# 处理数据
import pandas as pd
import matplotlib.pyplot as plt

def getdata():
    # df = pd.read_csv("../example_data/shanghai_1990-12-19_to_2019-2-28.csv")
    # df = pd.read_parquet("../data/battery_mean_soh.parquet", engine='pyarrow').dropna()
    df = pd.read_parquet("../data/mean_soh.parquet", engine='pyarrow').dropna()
    df = df.loc[df["device_id"] == "3e4a5515-fd4a-41cc-bd66-ed77f0f2ac2b", :]
    df = df.sort_values("date")
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

    return train,test

if __name__ == '__main__':
    getdata()