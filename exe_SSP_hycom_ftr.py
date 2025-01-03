import torch
import yaml
import os
from main_model_hycom_ftr import CSDI_SSP
from dataset_process_hycom_ftr import get_dataloader
from utils_hycom_ftr import train, evaluate

#代码运行顺序：generate，csvcreate，exe，pk2mat，pic.m

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
train(model,config["train"],show=config["train"]["show"],train_loader = train_loader,foldername=foldername)
#获取评估结果
evaluate(model, test_loader, nsample=config['evaluate']['nsample'], scaler=1, foldername=foldername)

