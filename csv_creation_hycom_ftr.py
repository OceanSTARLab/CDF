import pandas as pd
import csv
import os
import os.path as op
from data_generate_hycom_forecast_ftr import trainset
def creation(filepath1,filepath2,filepath3):
    headers = ['Time', 'Parameter', 'Value']
    parameter = ['Depth=0m','Depth=20m','Depth=40m','Depth=60m','Depth=80m',
                 'Depth=100m','Depth=120m','Depth=140m','Depth=160m','Depth=180m',
                 'Depth=200m','Depth=220m','Depth=240m','Depth=260m','Depth=280m',
                 'Depth=300m','Depth=320m','Depth=340m','Depth=360m','Depth=380m','Depth=400m']

    feature_num = len(parameter)
    save_train_and_test = []

    with open(filepath1) as f1:
        with open(filepath2) as f2:
            reader1 = csv.reader(f1)
            next(reader1)
            reader2 = csv.reader(f2)
            next(reader2)
            for i, j in zip(reader1, reader2):
                for k in range(feature_num):
                    save_train_and_test.append({'Time': str(int(j[0])), 'Parameter': parameter[k], 'Value': float(i[k])})

    with open(filepath3,'w',encoding='utf-8',newline='') as f3:
        writer = csv.DictWriter(f3,headers)
        writer.writeheader()
        writer.writerows(save_train_and_test)

filepath1 = './data/hycom_train_and_test_ftr.csv'
filepath2 = './data/TIME.csv'
filepath3 = './data/HYCOMtimeseries_Train_and_Test_ftr.csv'
creation(filepath1,filepath2,filepath3)