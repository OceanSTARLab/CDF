import pickle
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import data_generate_hycom_forecast_ftr
import yaml
from data_generate_hycom_forecast_ftr import trainset,mean,std,B,timepoint   #训练集长度,均值，标准差,样本数，每个样本的时间长度

path = "./config/base.yaml"
with open(path, "r",encoding='utf-8') as f:
    config = yaml.safe_load(f)
parameter_all = ['Depth=a','Depth=b','Depth=c','Depth=d','Depth=e','Depth=f','Depth=g',
             'Depth=h','Depth=i','Depth=j','Depth=k','Depth=l','Depth=m','Depth=n',
             'Depth=o','Depth=p','Depth=q','Depth=r','Depth=s','Depth=t','Depth=u']
start_depth = 0
end_depth = 20
parameter = parameter_all[start_depth:end_depth]
feature_sum = len(parameter_all)   #总共有21个深度
feature_num = len(parameter)       #我们选其中前20个
csvpath = "./data/HYCOMtimeseries_Train_and_Test_ftr.csv"
#生成观测值，验证集掩码以及观测值掩码矩阵。mask_mode包含两种模式，一种为'impute'，另一种为'predict'。
def data_and_mask_generate(missing_ratio,csvpath,mask_mode='predict'):
    data = pd.read_csv(csvpath)
    missing_num = int(timepoint * missing_ratio)
    observed_values = []
    testset_values = []
    for h in range(trainset):     #制作训练集
        observed_values.append(list(data['Value'][start_depth + h * feature_sum:end_depth + h * feature_sum]))
    observed_values = np.array(observed_values)
    observed_masks = np.ones(observed_values.shape)
    for h in range(trainset):     #制作测试集
        testset_values.append(list(data['Value'][start_depth+(h+missing_num)*feature_sum:end_depth+(h+missing_num)*feature_sum]))
    testset_values = np.array(testset_values)
    if mask_mode == 'impute':      #插补模式，随机掩码
        masks = observed_masks.reshape(-1).copy()     #为测试集掩码
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False)
        masks[miss_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)
    elif mask_mode == 'predict':   #预报模式，未来掩码
        masks = observed_masks.copy()        #为训练集掩码
        for index in range(trainset):
            if (index+1)%timepoint==0 or (index+1)%timepoint>timepoint-missing_num:
                masks[index] = np.zeros(masks[0].shape)
        gt_masks = masks     #得到掩码矩阵
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    return observed_values, testset_values, observed_masks, gt_masks

class SSP_Dataset(Dataset):
    def __init__(self, csvpath,timepoint = timepoint, use_index_list=None, missing_ratio=0.1):
        self.observed_values = []
        self.testset_values =[]
        self.observed_masks = []
        self.gt_masks = []
        self.obs_tes = []    #这是观测数据和测试数据的合体
        self.timepoint = timepoint
        path = ("./data/SSP_predict_" + "missing_ratio=" + str(missing_ratio) + ".pk")
        observed_values, testset_values, observed_masks, gt_masks= data_and_mask_generate(csvpath=csvpath,missing_ratio=missing_ratio)
        missing_num = int(timepoint * missing_ratio)
        for i in range(B):
            #为训练集和测试集掩码
            self.observed_values.append(observed_values[i * self.timepoint : (i + 1) * self.timepoint])
            self.testset_values.append(testset_values[i * self.timepoint : (i + 1) * self.timepoint] * gt_masks[i * self.timepoint:(i + 1) * self.timepoint])
            self.observed_masks.append(observed_masks[i * self.timepoint : (i + 1) * self.timepoint])
            self.gt_masks.append(gt_masks[i * self.timepoint : (i + 1) * self.timepoint])

        self.observed_values = np.array(self.observed_values)
        self.testset_values = np.array(self.testset_values)
        a = torch.tensor(self.observed_values)
        b = torch.tensor(self.testset_values)
        c = torch.cat((a,b))
        self.obs_tes = np.array(c)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)
        with open(path, "wb") as f:
            pickle.dump([self.observed_values, self.testset_values, self.gt_masks], f)
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.obs_tes))
        else:
            self.use_index_list = use_index_list
    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {"testset_data": self.testset_values[index],
             "observed_data": self.observed_values[index],
             "observed_mask": self.observed_masks[index],
             "gt_mask": self.gt_masks[index],
             "timepoints": np.arange(self.timepoint)}
        return s

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(batch_size, missing_ratio,csvpath):   #划分集
    dataset = SSP_Dataset(csvpath=csvpath,missing_ratio=missing_ratio)
    indlist = np.arange(len(dataset))
    num_train = B
    num_test = B
    train_index = indlist[:num_train]
    test_index = indlist[:num_test]
    dataset = SSP_Dataset(csvpath=csvpath,use_index_list=train_index, missing_ratio=missing_ratio)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=0)
    test_dataset = SSP_Dataset(csvpath=csvpath,use_index_list=test_index, missing_ratio=missing_ratio)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, test_loader
