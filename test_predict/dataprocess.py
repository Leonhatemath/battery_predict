# 处理数据
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np


def getdata(df):
    # df = pd.read_csv("../example_data/shanghai_1990-12-19_to_2019-2-28.csv")
    # df = pd.read_parquet("../data/battery_mean_soh.parquet", engine='pyarrow').dropna()
    df = df.loc[df["device_id"] == "384bcb86-ac9c-42da-a1c3-cd6e68186a06", :].dropna()
    df = df.sort_values("date")
    print(df.head())
    df["date"] = df["date"].map(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:])
    df.date = pd.to_datetime(df.date, format='%Y-%m-%d')
    df.index = df.date
    df["mean_soh"] = df["mean_soh"].astype(float).round(decimals=2)
    df.info()

    num = int(len(df) / 5 * 4)
    train = df[:num]
    train.index = train.date
    train.mean_soh.plot(figsize=(35, 8), title='Daily mean_soh', fontsize=6)
    print("train:")
    train.info()

    test = df[num:]
    test.index = test.date
    test.mean_soh.plot(figsize=(35, 8), title='Daily mean_soh', fontsize=6)
    print("test:")
    test.info()

    plt.show()

    return train, test


def getDescData(df):
    groupByDevice = df.groupby('device_id')
    deviceIds = {}
    for deviceId, group in groupByDevice:
        if group.shape[0] < 30:continue
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

    with open('../data/deviceIds.csv', 'w') as f:
        w = csv.writer(f)
        for key, value in deviceIds.items():
            w.writerow([key, value])

    f.close()

def upOrDown(x, y):
    a, b = np.polyfit(x,y,1)
    return a < 0

if __name__ == '__main__':
    df = pd.read_parquet("../data/mean_soh.parquet", engine='pyarrow').dropna()
    getdata(df)
    # getDescData(df)
