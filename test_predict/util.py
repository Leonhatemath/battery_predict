from sklearn.metrics import mean_squared_error
from math import sqrt

def acc_count(test,y_hat):
    rms = sqrt(mean_squared_error(test.mean_soh, y_hat.value))
    return rms