import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error # 常用的R方、均方根误差 

svr_runoff = SVR()
forest_runoff = RandomForestRegressor(random_state=42, n_jobs=-1)
BP = MLPRegressor(learning_rate='adaptive', max_iter=2000,random_state=42)

all_estimators = [
    ('svr', svr_runoff),
    ('forest', forest_runoff),
    ('BP', BP)
    ]

def MAPE(y_true, y_pred): # 平均绝对百分比误差
    return np.mean(np.abs((y_true - y_pred) / y_true))

def evalue(y_pred, y_true): # 评估指标r2 
    r2 = r2_score(y_pred=y_pred, y_true=y_true)
    rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
    mape = MAPE(y_pred=y_pred, y_true=y_true) 
    return {"r2": r2, "rmse": rmse, "mape": mape}