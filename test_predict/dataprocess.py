# 处理数据
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from test_predict.util import *


def getdata(df, deviceId, model):
    # df = pd.read_csv("../example_data/shanghai_1990-12-19_to_2019-2-28.csv")
    # df = pd.read_parquet("../data/battery_mean_soh.parquet", engine='pyarrow').dropna()
    if (model is None):
        df = df.loc[df["device_id"] == deviceId, :].dropna().loc[:, ['date', 'mean_soh']]
    else:
        df = df.loc[df["model"] == model, :].dropna().loc[:, ['date', 'mean_soh']]
    df = df.sort_values("date")
    df["date"] = df["date"].map(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:])
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df.index = df.date
    df["mean_soh"] = df["mean_soh"].astype(float).round(decimals=2)
    print(df.head())
    df.info()

    num = int(df.shape[0] / 5 * 4)
    train = df[:num]
    train.index = train.date
    tra_end = train.index[-1]
    train.mean_soh.plot(figsize=(35, 8), title='Daily mean_soh', fontsize=6)
    print("train:")
    train.info()

    test = df[num:]
    test.index = test.date
    tes_end = df.shape[0] - num
    test.mean_soh.plot(figsize=(35, 8), title='Daily mean_soh', fontsize=6)
    print("test:")
    test.info()

    plt.show()

    return train, test, tra_end, tes_end

def getdata(df, model):
    # df = pd.read_csv("../example_data/shanghai_1990-12-19_to_2019-2-28.csv")
    # df = pd.read_parquet("../data/battery_mean_soh.parquet", engine='pyarrow').dropna()
    df = df.loc[df["model"] == model, :].dropna().loc[:, ['cyclecount', 'mean_soh']]
    df = df.sort_values("cyclecount")
    df.index = df.cyclecount
    df["mean_soh"] = df["mean_soh"].astype(float).round(decimals=2)
    print(df.head())
    df.info()

    num = int(df.shape[0] / 5 * 4)
    train = df[:num]
    train.index = train.cyclecount
    tra_end = train.index[-1]
    train.mean_soh.plot(figsize=(35, 8), title='Daily mean_soh', fontsize=6)
    print("train:")
    train.info()

    test = df[num:]
    test.index = test.cyclecount
    tes_end = df.shape[0] - num
    test.mean_soh.plot(figsize=(35, 8), title='Daily mean_soh', fontsize=6)
    print("test:")
    test.info()

    plt.show()

    return train, test, tra_end, tes_end

def getDescData(df):
    groupByDevice = df.groupby('device_id')
    deviceIds = {}
    for deviceId, group in groupByDevice:
        if group.shape[0] < 30: continue
        group = group.sort_values('date', ascending=True)
        soh = list(group['mean_soh'])
        model = group.iloc[0, 0]
        x = list(map(int, group['date']))
        if upOrDown(x, soh):
            if not (model in deviceIds.keys()):
                deviceIds[model] = []
            deviceIdList = deviceIds.get(model)
            deviceIdList.append(deviceId)
            deviceIds[model] = deviceIdList

    with open('../data/20201220_deviceIds.csv', 'w') as f:
        w = csv.writer(f)
        for key, value in deviceIds.items():
            w.writerow([key, value])

    f.close()


def upOrDown(x, y):
    a, b = np.polyfit(x, y, 1)
    return a < 0


if __name__ == '__main__':
    # 比较好得一些数据：["00cc4087-af9b-4cfa-a378-c0ced676b44e",]
    df = pd.read_parquet("../data/20201221soh_by_date.parquet", engine='pyarrow').dropna()
    df1 = pd.read_parquet("../data/20201221soh_by_cyclecount.parquet",engine='pyarrow').dropna()
    # deviceId = "06ccf344-65b5-4a5b-9934-dea3152b2182"
    # train, test, tra_end, tes_end = getdata(df, deviceId, None)
    model = 'Mi 10 Pro'
    # train, test, tra_end, tes_end = getdata(df, None, model)
    train, test, tra_end, tes_end = getdata(df1, model)
    # shift_train = data_to_supervised(train)
    # print(shift_train.head())
    # getDescData(df)
