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
df = pd.read_parquet("../data/20201221soh_by_date.parquet", engine='pyarrow').dropna()
df1 = pd.read_parquet("../data/20201221soh_by_cyclecount.parquet", engine='pyarrow').dropna()
# deviceId = "00cc4087-af9b-4cfa-a378-c0ced676b44e"
model = 'Mi 10 Pro'
# train, test, tra_end, tes_end = getdata(df, None, model)
train, test, tra_end, tes_end = getdata(df1, model)
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
    fit = ExponentialSmoothing(np.asarray(train[count]), seasonal_periods=2, trend='add', seasonal='add').fit()
    predict[value] = fit.forecast(len(test))
    return train, test, predict


def ARIMA():
    # d = getD(train.mean_soh)
    # resDiff = sm.tsa.arma_order_select_ic(train.mean_soh, max_ar=7, max_ma=7, ic='aic', trend='c')
    # p = resDiff['aic_min_order'][0]
    # q = resDiff['aic_min_order'][1]
    fit = sm.tsa.statespace.SARIMAX(train['mean_soh'], order=(5, 1, 2),
                                    enforce_invertibility=False, enforce_stationarity=False).fit()
    predict[value] = fit.predict(start=tra_end, dynamic=True)[1:tes_end]
    return train, test, predict

def getD(meanSoh):
    res = 0
    while sm.tsa.adfuller(meanSoh)[0] >= sm.tsa.adfuller(meanSoh)[4]['5%']:
        meanSoh = np.diff(meanSoh)
        res += 1
    return res



def predict_method(way):
    operator = {'native_way': native_way,
                'native_avg': native_avg,
                'move_avg': move_avg,
                'ses': ses,
                'holtLiner': holtLiner,
                'holtWinter': holtWinter,
                'ARIMA': ARIMA}
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
    # 霍尔特线性趋势：holtLiner;霍尔特季节性预测模型：holtWinter;自回归移动平均模型：ARIMA
    way = "holtLiner"
    time_series_predict(way)
