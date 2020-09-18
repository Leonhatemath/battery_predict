# 比特币走势预测，使用时间序列 ARMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
# 数据加载
df = pd.read_csv(r'C:\Users\mi\Documents\python\data analysis\bitcoin-master\bitcoin_2012-01-01_to_2018-10-31.csv')
# 将时间作为 df 的索引
df.Timestamp = pd.to_datetime(df.Timestamp)
df.index = df.Timestamp
# 数据探索
#print(df.head())
# 按照月，季度，年来统计
df_month = df.resample('M').mean()
df_Q = df.resample('Q-DEC').mean()
df_year = df.resample('A-DEC').mean()
# 按照天，月，季度，年来显示比特币的走势
plt.figure(figsize=[15, 7])
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.suptitle('比特币金额（美金）', fontsize=20)
plt.subplot(221)
plt.plot(df.Weighted_Price, '-', label='按天')
plt.legend()
plt.subplot(222)
plt.plot(df_month.Weighted_Price, '-', label='按月')
plt.legend()
plt.subplot(223)
plt.plot(df_Q.Weighted_Price, '-', label='按季度')
plt.legend()
plt.subplot(224)
plt.plot(df_year.Weighted_Price, '-', label='按年')
plt.legend()
plt.show()
#根据AIC准则选择最佳阶数
ps = range(0,4)
parameters_list = list(product(ps,ps))  #返回ps和ps中的元素组成的笛卡尔积的元组
best_aic = float('inf')
result = []
for parameters in parameters_list:
    try:
        model = ARMA(df_month.Weighted_Price,order = (parameters[0],parameters[1])).fit()
    except ValueError:
        print('参数错误：',parameters)
        continue
    aic = model.aic
    if aic< best_aic:
        best_aic = aic
        best_model = model
        best_parameters = parameters
    result.append([parameters,model.aic])
#输出最优模型
result_table = pd.DataFrame(result)
result_table.columns = ['parameter','aic']
print('最优模型：',best_model.summary())
#结果预测
df_month2 = df_month[['Weighted_Price']]
future = pd.DataFrame(index = [datetime(2018,11,30),datetime(2018,12,31),datetime(2019,1,31),
                               datetime(2019,2,28),datetime(2019,3,31),datetime(2019,4,30),
                               datetime(2019,5,31),datetime(2019,6,30)],columns = df_month.columns)
df_month2 = pd.concat([df_month2,future])
df_month2['forecast'] = best_model.predict(start = 0,end = 91)
#结果可视化
plt.figure(figsize = (15,7))
df_month2.Weighted_Price.plot(label = '实际金额')
df_month2.forecast.plot(color = 'r',ls = '--',label= '预测金额')
plt.legend()
plt.title('比特币金额（月）')
plt.xlabel('时间')
plt.ylabel('金额')
plt.show()
