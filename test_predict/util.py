from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame
from pandas import concat

def acc_count(test,y_hat):
    rms = sqrt(mean_squared_error(test.mean_soh, y_hat.value))
    return rms

def data_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    """
    	将时间序列重构为监督学习数据集.
    	参数:
    		data: 观测值序列，类型为列表或Numpy数组。
    		n_in: 输入的滞后观测值(X)长度。
    		n_out: 输出观测值(y)的长度。
    		dropnan: 是否丢弃含有NaN值的行，类型为布尔值。
    	返回值:
    		经过重组后的Pandas DataFrame序列.
    	"""
    names = ["date", "mean_soh"]
    cols = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (names[j], i)) for j in range(len(names))]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (names[j])) for j in range(len(names))]
        else:
            names += [('%s(t+%d)' % (names[j], i)) for j in range(len(names))]
    # 将列名和数据拼接在一起
    agg = concat(cols, axis=1)
    agg.columns = names
    # 丢弃含有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg