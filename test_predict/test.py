import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_parquet("../data/old_mean_soh.parquet", engine='pyarrow')
df = df.dropna()
df.info()
df = df.loc[df["device_id"] == '3e4a5515-fd4a-41cc-bd66-ed77f0f2ac2b', ['date','mean_soh','std']]
print(df[:10])
