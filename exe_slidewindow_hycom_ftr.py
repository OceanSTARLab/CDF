import torch
import yaml
import os
from main_model_hycom_ftr import CSDI_SSP
from dataset_process_hycom_ftr import get_dataloader
from utils_hycom_ftr import train, evaluate

import pickle
import pandas as pd
import numpy as np
import scipy
import scipy.io as sio
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

#代码运行顺序：generate，csvcreate，exe，pk2mat，pic.m

#这是第一轮的数据
csvpath = "./data/HYCOMtimeseries_Train_and_Test_ftr.csv"

path = "./config/base.yaml"
with open(path, "r",encoding='utf-8') as f:
    config = yaml.safe_load(f)
foldername = "./save/SSPhycom预报数据"
os.makedirs(foldername, exist_ok=True)
#获取训练集，验证集，以及测试集数据
train_loader, test_loader = get_dataloader(batch_size=config["train"]["batch_size"],missing_ratio=config["model"]["missing_ratio"],csvpath=csvpath)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = CSDI_SSP(config, device).to(device)
train(model,config["train"],train_loader = train_loader,show=config["train"]["show"],foldername=foldername)
#获取评估结果
evaluate(model, test_loader, nsample=config['evaluate']['nsample'], scaler=1, foldername=foldername)

import data_generate_hycom_forecast_ftr
from data_generate_hycom_forecast_ftr import trainset,mean,std,B,timepoint,test_B,feature_num,missing_num,normalize,csv_generate,month
import csv_creation_hycom_ftr
from csv_creation_hycom_ftr import creation

#总共ftr轮滑窗预报（含首轮，相当于做ftr-1轮滑窗预报）
ftr = config['evaluate']['ftr']

#死变量
filename = './data/hycomdownloaddata/2022/year2022month'+str(month)+'.mat'
data_all = missing_num * ftr + timepoint
database = sio.loadmat(filename)  # database是字典类型,字典的键为ssf
data = database['ssf']
data = torch.tensor(data)  # data.shape = (248,38,76,21)
data = data[:, :, 0, :]  # data.shape = (248,38,21)
data = data.permute(1, 2, 0)  # data.shape = (38,21,248)
data = data[:B, :, :data_all]  # data.shape = (B,21,data_all)
p = data[0]
for i in range(1, B):
    p = torch.cat((p, data[i]), dim=1)
data = np.array(p).T
data = data[trainset + B * missing_num:]  # 存放测试集数据
data = data.tolist()

#正在获取后续轮次评测数据
for turn in range(1,ftr):
    #获取上一轮的预测数据
    if turn == 1:
        pre = sio.loadmat("./save/SSPhycom预报数据/forecasting_results.mat")
    else:
        pre = sio.loadmat(f'./save/SSPhycom预报数据/forecasting_results_ftr={turn-1}.mat')
    prelst = pre['data']    #此时p是列表,shape(120,20)
    #需要补充0
    prelst=np.array(prelst).T
    b = prelst.shape
    prelen = b[1]
    all0 = [0]*prelen
    prelst = prelst.tolist()
    prelst.append(all0)
    prelst = np.array(prelst)
    #上一轮的预测数据需要归一化
    normalize(prelst,prelen,mean,std)
    prelst = prelst.T.tolist()     #120,21
    #获取原始数据，后续将原始数据与历轮预测数据拼接
    if turn == 1:
        org = sio.loadmat("./data/hycom_train_and_test_ftr.mat")
        r = org['ssf']
    else:
        org = sio.loadmat(f"./save/滑窗预报中间数据文件夹/hycom_train_and_test_ftr={turn-1}.mat")
        r = np.array(org['ssf']).T.tolist()
    r = r[:trainset]  # 1440,21
    #开始制作下一轮的数据
    d1 = []
    for i in range(B):
        d1.extend(r[i*timepoint+missing_num:(i+1)*timepoint])     #训练集
        d1.extend(prelst[i*missing_num:(i+1)*missing_num])        #测试集
    #计算误差用,获取理论值
    for i in range(B):
        d1.extend(data[(turn-1)*missing_num+i*(ftr-1)*missing_num:(turn-1)*missing_num+i*(ftr-1)*missing_num+missing_num])
    d1 = np.array(d1)
    sio.savemat(f'./save/滑窗预报中间数据文件夹/hycom_train_and_test_ftr={turn}.mat', {'ssf': d1.T})
    features = sio.loadmat(f'./save/滑窗预报中间数据文件夹/hycom_train_and_test_ftr={turn}.mat')['ssf'].T
    dfdata = pd.DataFrame(features)
    csv_train_data_path = f'./save/滑窗预报中间数据文件夹/hycom_train_and_test_ftr={turn}.csv'
    dfdata.to_csv(csv_train_data_path, index=False)
    filepath1 = csv_train_data_path
    filepath2 = './data/TIME.csv'
    filepath3 = f'./save/滑窗预报中间数据文件夹/HYCOMtimeseries_Train_and_Test_ftr={turn}.csv'
    creation(filepath1,filepath2,filepath3)
    #这是第turn轮的数据
    csvpath_turn = f"./save/滑窗预报中间数据文件夹/HYCOMtimeseries_Train_and_Test_ftr={turn}.csv"
    #获取第turn轮数据
    train_loader, test_loader = get_dataloader(batch_size=config["train"]["batch_size"],missing_ratio=config["model"]["missing_ratio"],csvpath=csvpath_turn)
    #获取评估结果(第turn轮)
    evaluate(model, test_loader, nsample=config['evaluate']['nsample'], scaler=1, foldername=foldername, ftr=turn)

error_mae = []
error_rmse = []
with open("./save/SSPhycom预报数据/error.pk",'rb') as file:
    a = pickle.load(file)
error_mae.append(a[0].tolist())
error_rmse.append(a[1].tolist())
for i in range(1,8):
    with open(f"./save/SSPhycom预报数据/error_ftr={i}.pk", 'rb') as file:
        a = pickle.load(file)
        error_mae.append(a[0].tolist())
        error_rmse.append(a[1].tolist())

print(error_mae)
datamae = {'mae':error_mae}
datarmse = {'rmse':error_rmse}

sio.savemat('./save/SSPhycom预报数据/error_mae.mat',datamae)
sio.savemat('./save/SSPhycom预报数据/error_rmse.mat',datarmse)