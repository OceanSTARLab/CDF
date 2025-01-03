import os
import os.path as op
import scipy
import scipy.io as sio
import pandas as pd
import numpy as np
import yaml
import torch

parameter = "abcdefghijklmnopqrstu"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

path = "./config/base.yaml"
with open(path, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)
missing_ratio = config["model"]["missing_ratio"]
#用哪个月份的数据做实验
month = 6

#本实验要预报未来
B = 30   #选取B个纬度latitude作为训练集(和验证集)
test_B = B    #测试集一共B个，因为我们要预报所有训练集的未来声速
timepoint = 48
trainset = B * timepoint    #训练集+验证集
missing_num = int(timepoint*missing_ratio)      #遮蔽部分的点数
data_all = timepoint+missing_num    #训练集和测试集一共这么多
is_normalize = 1    #归一化
feature_num = len(parameter)
def normalize(data,trainset,mean,std):
    data[:,:trainset] = (data[:,:trainset] - mean) / std
    return data
def generate(timepoint,B):
    #导入原始数据集并处理。在本案例中，经度为样本维度。
    filename = './data/hycomdownloaddata/2022/year2022month'+ str(month) +'.mat'
    database = sio.loadmat(filename)  # database是字典类型,字典的键为ssf
    data = database['ssf']
    data = torch.tensor(data)   #data.shape = (248,38,76,21)
    data = data[:,:,0,:]        #data.shape = (248,38,21)
    data = data.permute(1,2,0)  #data.shape = (38,21,248)
    data = data[:B,:,:data_all]    #data.shape = (B,21,data_all)
    p = data[0]
    for i in range(1,B):
        p = torch.cat((p,data[i]),dim=1)
    data = np.array(p)
    mean = np.mean(data[:trainset],axis=1)
    std = np.std(data[:trainset],axis=1)
    mean = mean.reshape((-1,1))
    std = std.reshape((-1,1))
    #归一化
    if is_normalize == 1:
        data = normalize(data,trainset,mean,std)
        data = data.T
    convey(data)
    csv_generate()
    return mean,std

def convey(data):
    sio.savemat('./data/hycom_train_and_test_ftr.mat', {'ssf': data})
def csv_generate():
    features = sio.loadmat('./data/hycom_train_and_test_ftr.mat')['ssf'].T
    dfdata = pd.DataFrame(features).T
    csv_train_data_path = './data/hycom_train_and_test_ftr.csv'
    dfdata.to_csv(csv_train_data_path, index=False)
mean,std = generate(timepoint,B)