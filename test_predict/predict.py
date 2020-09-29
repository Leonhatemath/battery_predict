from dataprocess import getdata
from util import acc_count
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
import pandas as pd

# 用前一天的数据作为下一天的预测数据

count = 'mean_soh'
Train = 'Train'
Test = 'Test'
value = 'value'
df = pd.read_parquet("../data/mean_soh.parquet", engine='pyarrow').dropna()
train, test = getdata(df)
predict = test.copy()


# 前一天的数据作为下一天的预测
def native_way():
    dd = np.asarray(train.mean_soh)
    predict[value] = dd[len(dd) - 1]
    return train, test, predict


def native_avg():
    predict[value] = train.mean_soh.mean()
    return train, test, predict


def move_avg():
    predict[value] = train.mean_soh.rolling(30).mean().iloc[-1]
    return train, test, predict


def ses():
    fit = SimpleExpSmoothing(np.asarray(train[count])).fit(smoothing_level=0.6, optimized=False)
    predict[value] = fit.forecast(len(test))
    return train, test, predict


def holtLiner():
    fit = Holt(np.asarray(train[count])).fit(smoothing_level=0.3, smoothing_trend=0.1)
    predict[value] = fit.forecast(len(test))
    return train, test, predict


def holtWinter():
    fit = ExponentialSmoothing(np.asarray(train[count]), seasonal_periods=1, trend='add', seasonal='add').fit()
    predict[value] = fit.forecast(len(test))
    return train, test, predict


def SARIMA():
    fit = sm.tsa.statespace.SARIMAX(train.mean_soh, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
    predict[value] = fit.predict(start="2020-08-26", end='2020-09-18', dynamic=True)
    return train, test, predict

def predict_method(way):
    operator = {'native_way': native_way,
                'native_avg': native_avg,
                'move_avg': move_avg,
                'ses': ses,
                'holtLiner': holtLiner,
                'holtWinter': holtWinter,
                'SARIMA': SARIMA}
    func = operator.get(way, lambda: "Invalid method")
    return func()


def time_series_predict(way):
    train, test, predict = predict_method(way)
    plt.figure(figsize=(35, 8))
    plt.plot(train[count], label=Train)
    plt.plot(test[count], label=Test)
    plt.plot(predict[value], label=way)
    plt.title(way)
    plt.show()
    print("均方根误差为：", acc_count(test, predict))


if __name__ == '__main__':
    # 朴素法：native_way;简单平均：native_avg;移动平均：move_avg;简单指数平滑法：ses;
    # 霍尔特线性趋势：holtLiner;霍尔特季节性预测模型：holtWinter;自回归移动平均模型：SARIMA
    way = "SARIMA"
    time_series_predict(way)
