import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import yaml
import os
import scipy
import scipy.io as sio
import math as m
import matplotlib.pyplot as plt
import pandas as pd
from data_generate_hycom_forecast_ftr import mean,std,is_normalize,trainset,feature_num,timepoint,missing_num,test_B
from dataset_process_hycom_ftr import start_depth,end_depth

parameter_all = ['Depth=a','Depth=b','Depth=c','Depth=d','Depth=e','Depth=f','Depth=g',
             'Depth=h','Depth=i','Depth=j','Depth=k','Depth=l','Depth=m','Depth=n',
             'Depth=o','Depth=p','Depth=q','Depth=r','Depth=s','Depth=t','Depth=u']
show = 10
path = "./config/base.yaml"
saveplace = "./save/SSPhycom预报数据/"
path_tes = "./data/hycom_train_and_test_ftr.csv"
with open(path, "r",encoding='utf-8') as f:
    config = yaml.safe_load(f)
missing_ratio = config["model"]["missing_ratio"]
mean = mean[start_depth:end_depth]
std = std[start_depth:end_depth]

def train(model,config,train_loader,show,foldername=""):
    optimizer = Adam(model.parameters(), lr=0.002, weight_decay=1e-6)
    #学习率在指定的迭代次数下降十倍（gamma=0.1）
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        j = 1
        model.train()   #进入训练模式，有用但不多
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):   #batch_no是batch的序号，train_batch是这个batch
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                if j%show ==0:
                    it.set_postfix(ordered_dict={"avg_epoch_loss": avg_loss / batch_no,"epoch": epoch_no+1},refresh=False)
                j=j+1
            lr_scheduler.step()

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)))
def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=mean,std_scaler = std, foldername="",ftr=0):
    with torch.no_grad():
        model.eval()
        prediction = []
        minsample = []
        maxsample = []
        if ftr==0:
            data = pd.read_csv("./data/hycom_train_and_test_ftr.csv")
            testdata = data[-missing_num * test_B:]  # 测试集原始数据
            testdata = np.array(testdata)
            testdata = torch.tensor(testdata)
            e = testdata[0][start_depth:end_depth].reshape((1, -1))
            for i in range(1, len(testdata)):
                e = torch.cat((e, testdata[i][start_depth:end_depth].reshape((1, -1))))
            target = e
        else:
            data = pd.read_csv(f"./save/滑窗预报中间数据文件夹/hycom_train_and_test_ftr={ftr}.csv")
            testdata = data[-missing_num * test_B:]  # 测试集原始数据
            testdata = np.array(testdata)
            testdata = torch.tensor(testdata)
            e = testdata[0][start_depth:end_depth].reshape((1, -1))
            for i in range(1, len(testdata)):
                e = torch.cat((e, testdata[i][start_depth:end_depth].reshape((1, -1))))
            target = e
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                samples_median = samples.quantile(0.5,dim=1)
                samples_min = samples.quantile(0.01,dim=1)
                samples_max = samples.quantile(0.99,dim=1)
                # 选择采样的中间点作为我们需要的值。
                prediction.append(samples_median)
                minsample.append(samples_min)
                maxsample.append(samples_max)
            a = prediction[0][0][-missing_num:].cpu() * std.T + mean.T
            b = maxsample[0][0][-missing_num:].cpu() * std.T + mean.T
            c = minsample[0][0][-missing_num:].cpu() * std.T + mean.T
            for j in range(1,test_B):
                a=torch.cat((a,prediction[j][0][-missing_num:].cpu() * std.T + mean.T))
                b=torch.cat((b,maxsample[j][0][-missing_num:].cpu()*std.T + mean.T))
                c=torch.cat((c,minsample[j][0][-missing_num:].cpu()*std.T + mean.T))
            gap = abs(target.cpu()-a.cpu())
            mae = sum(gap.cpu())/(missing_num*test_B)
            rmse = (sum(gap.cpu()**2)/(missing_num*test_B))**0.5
            print(f"The RMSE of turn {ftr+1} is:")
            print(rmse)
            print(f"The MAE of turn {ftr+1} is:")
            print(mae)

    #存放数据（首轮不做额外标记）
    if ftr==0:
        with open(saveplace+"forecasting_results.pk",'wb') as f:
            pickle.dump(a,f)
        with open(saveplace+"max_results.pk",'wb') as f:
            pickle.dump(b,f)
        with open(saveplace+"min_results.pk",'wb') as f:
            pickle.dump(c,f)
        with open(saveplace+"error.pk",'wb') as f:
            pickle.dump([mae,rmse],f)
    else:
        with open(saveplace+"forecasting_results_ftr="+ str(ftr) +".pk",'wb') as f:
            pickle.dump(a,f)
        with open(saveplace+"max_results_ftr="+ str(ftr) +".pk",'wb') as f:
            pickle.dump(b,f)
        with open(saveplace+"min_results_ftr="+ str(ftr) +".pk",'wb') as f:
            pickle.dump(c,f)
        with open(saveplace+"error_ftr="+ str(ftr) +".pk",'wb') as f:
            pickle.dump([mae,rmse],f)

    #制作.mat文件
    prediction = []
    targetlist = []
    maxlist = []
    minlist = []
    if ftr==0:
        with open("./save/SSPhycom预报数据/forecasting_results.pk", 'rb') as file:
            a = pickle.load(file)
        with open("./save/SSPhycom预报数据/max_results.pk", 'rb') as file:
            b = pickle.load(file)
        with open("./save/SSPhycom预报数据/min_results.pk", 'rb') as file:
            c = pickle.load(file)
        for j in range(int(missing_num * test_B)):
            prediction.append(a[j][:][:].tolist())
            targetlist.append(target[j][:][:].tolist())
            maxlist.append(b[j][:][:].tolist())
            minlist.append(c[j][:][:].tolist())
        data = {'data': prediction}
        data1 = {'data1': targetlist}
        data2 = {'max': maxlist}
        data3 = {'min': minlist}
        sio.savemat('./save/SSPhycom预报数据/forecasting_results.mat', data)
        sio.savemat('./save/SSPhycom预报数据/target.mat', data1)
        sio.savemat('./save/SSPhycom预报数据/max_results.mat', data2)
        sio.savemat('./save/SSPhycom预报数据/min_results.mat', data3)
    else:
        with open("./save/SSPhycom预报数据/forecasting_results_ftr="+ str(ftr) +".pk", 'rb') as file:
            a = pickle.load(file)
        with open("./save/SSPhycom预报数据/max_results_ftr="+ str(ftr) +".pk", 'rb') as file:
            b = pickle.load(file)
        with open("./save/SSPhycom预报数据/min_results_ftr="+ str(ftr) +".pk", 'rb') as file:
            c = pickle.load(file)
        for j in range(int(missing_num * test_B)):
            prediction.append(a[j][:][:].tolist())
            targetlist.append(target[j][:][:].tolist())
            maxlist.append(b[j][:][:].tolist())
            minlist.append(c[j][:][:].tolist())
        data = {'data': prediction}
        data1 = {'data1': targetlist}
        data2 = {'max': maxlist}
        data3 = {'min': minlist}
        sio.savemat('./save/SSPhycom预报数据/forecasting_results_ftr='+ str(ftr) +'.mat', data)
        sio.savemat('./save/SSPhycom预报数据/target_ftr='+ str(ftr) +'.mat', data1)
        sio.savemat('./save/SSPhycom预报数据/max_results_ftr='+ str(ftr) +'.mat', data2)
        sio.savemat('./save/SSPhycom预报数据/min_results_ftr='+ str(ftr) +'.mat', data3)