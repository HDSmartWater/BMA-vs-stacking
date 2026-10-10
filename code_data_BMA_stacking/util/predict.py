import numpy as np

def predict_with_error(model, use_day, predict_day, x_train, x_test):
    max_predict_day = max(predict_day)
    test_len = len(x_test)
    new_X = np.concatenate((x_train[-(max_predict_day - 1):], x_test), axis=0)
    pre_n = [[] for i in range(max_predict_day)]
    for n in range(max_predict_day): # n记录预测预见期
        X = new_X[n:]
        if n < use_day and n:
            t = np.array([p[-len(X):] for p in pre_n[0:n]])
            t = np.transpose(t)
            X[:, use_day-n:use_day] = t
        elif n >= use_day:
            t = np.array([p[-len(X):] for p in pre_n[n-use_day:n]])
            t = np.transpose(t)
            X[:, 0:use_day] = t
        pre_n[n] = model.predict(X)
    return_list = []
    for need in predict_day:
        return_list.append(pre_n[need-1][-test_len:])
    return return_list

# from data_process import process
# from estimators import all_estimators, eval
# from grid_search import search_bestmodel

# class config():
#     streamflow_path = './data/camels_streamflow.nc'
#     forcing_data_path = './data/camels_forcingmete.nc'
#     # 选定前n天的驱动因素为特征
#     n = 7
#     # 预测天数
#     id = '01013500'
#     predict_day = [1, 3, 7, 15, 30]
#     outputpath = './outputs'
#     #  选取开始时间
#     start_date='2000-01-01'
#     #  选取结束时间
#     end_date='2001-01-01'

# x_train, y_train, x_test, y_test ,norm_param = process(train_size=0.8, useday_n=config.n, selected_id=config.id, start_date=config.start_date, end_date=config.end_date,
#                                             predict_day=1, streamflow_path=config.streamflow_path, forcing_data_path=config.forcing_data_path)
# elastic = search_bestmodel(all_estimators[3][1], x_train, y_train, all_estimators[3][0], '1')
# predict_with_error(model=elastic, use_day=config.n, predict_day=config.predict_day, x_train=x_train, x_test=x_test)
