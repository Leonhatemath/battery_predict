import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet("../data/battery_mean_soh.parquet", engine='pyarrow')
df = df.dropna()
df.info()
df = df.loc[df["days"] == 172, ["model", "device_id"]]
print(df[:10])
