# -*- coding: utf-8 -*-

# 用 ARMA 进行时间序列预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA

# 创建数据
data = pd.Series([5922, 5308, 5546, 5975, 2704, 1767, 4111, 5542, 4726, 5866, 6183, 3199, 1471,
                  1325, 6618, 6644, 5337, 7064, 2912, 1456, 4705, 4579, 4990, 4331, 4481, 1813,
                  1258, 4383, 5451, 5169, 5362, 6259, 3743, 2268, 5397, 5821, 6115, 6631, 6474,
                  4134, 2728, 5753, 7130, 7860, 6991, 7499, 5301, 2808, 6755, 6658, 7644, 6472,
                  8680, 6366, 5252, 8223, 8181, 10548, 11823, 14640, 9873, 6613, 14415, 13204,
                  14982, 9690, 10693, 8276, 4519, 7865, 8137, 10022, 7646, 8749, 5246, 4736, 9705,
                  7501, 9587, 10078, 9732, 6986, 4385, 8451, 9815, 10894, 10287, 9666, 6072, 5418])
data.index = pd.Index(sm.tsa.datetools.dates_from_range('1901', '1990'))
# 绘制数据图
data.plot(figsize=(12, 8))
plt.show()

# 根据AIC准则自动定阶
# (p, q) =(sm.tsa.arma_order_select_ic(data,max_ar=10,max_ma=10,ic='aic')['aic_min_order'])
# print((p,q))
# 根据自相关图、偏自相关图定阶
plot_acf(data)  # 自相关图
plot_pacf(data)  # 偏自相关图
plt.show()
r, rac, Q = sm.tsa.acf(data, qstat=True)
prac = pacf(data, method='ywmle')
table_data = np.c_[range(1, len(r)), r[1:], rac, prac[1:len(rac) + 1], Q]
table = pd.DataFrame(table_data, columns=['lag', "AC", "Q", "PAC", "Prob(>Q)"])
print(table)

# 模型结果生成
p, d, q = (7, 0, 7)
arma_mod = ARMA(data, (p, d, q)).fit(disp=-1, method='mle')
summary = (arma_mod.summary2(alpha=.05, float_format="%.8f"))
print(summary)

# 创建 ARMA 模型 # 创建 ARMA 模型
arma = ARMA(data, (7, 0)).fit()
print('AIC: %.4lf' % arma.aic)
# 模型预测
predict_y = arma.predict('1990', '2000')
# 预测结果绘制
fig, ax = plt.subplots(figsize=(12, 8))
ax = data.index['1901':].plot(ax=ax)
predict_y.plot(ax=ax)
plt.show()
