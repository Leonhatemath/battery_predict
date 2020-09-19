# 处理数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv("../example_data/shanghai_1990-12-19_to_2019-2-28.csv")
df = pd.read_parquet("../data/battery_mean_soh.parquet", engine='pyarrow')
df.info()
df = df.loc[df["device_id"] == "e27b273a-3207-4355-9b04-9b1786981f0e", :]
df = df.sort_values("date")
print(df.loc[:, ["model", "date", "mean_soh"]])
train = df[:130]
test = df[130:]
train.Timestamp = pd.to_datetime(train["Timestamp"], format='%y-%m-%d')
train.index = train.Timestamp
test.Timestamp = pd.to_datetime(test["Timestamp"], format='%y-%m-%d')
test.index = test.Timestamp
