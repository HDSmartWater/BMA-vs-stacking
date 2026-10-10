import pandas as pd
import numpy as np
import xarray as xr

def dataset_split(dataset, begin, end, useday_n, predict_day): #时序序列不需要打乱
    """
    数据集分割 dataset_split，用于从完整的数据中分割出训练集或测试集
        Input输入：
            dataset：nparray，输入所有的数据。
            begin, end：int，分别对应着分割开始时间和结束时间的序列号。
            useday_n: int，特征使用的天数。
            predict_day：int，预测的天数。
        Output输出：
            x_set, y_set：nparray，输入对应的特征和标签。
    """
    # 创建空的nparray,长度为使用数据天数*特征数
    x_set = np.empty((0, len(dataset) * useday_n))
    for b in range(begin, end - useday_n):
        x_traindatanew = np.array([d[b: b + useday_n] for d in dataset]).reshape(1, -1)
        x_set = np.concatenate((x_set, x_traindatanew), axis=0)
    # 追加未来 predict_day 天的气象驱动因子作为特征（不含径流，假设气象预报已知）
    for i in range(1, len(dataset)):
        for d in range(predict_day):
            newdata = dataset[i][useday_n + d + begin: end + d]
            x_set = np.concatenate((x_set, newdata.reshape(-1, 1)), axis=1)
    y_set = np.empty((end - begin - useday_n , 0))
    for i in range(predict_day-1, predict_day):
        nextday_y = dataset[0][useday_n + begin + i: end + i].reshape(-1, 1)
        y_set = np.concatenate((y_set, nextday_y), axis=1)
    return x_set, y_set


def nomarlization_01(data_set, begin, end):
    """
    数据集归一化,零一归一化并返回归一化参数
        Input输入:
            data_set:nparray,输入所有的数据（径流、降水、最高气温、最低气温）。
            begin, end:int,归一化统计量（max/min）所取的数据区间序号，即训练段，防止信息泄漏。
        Output输出:
            data_normalized:nparray,输入对应的特征和标签。
            norm_param:dict,归一化参数列表。  norm_param 为{'max':[max0, max1, ..., maxn],'min':[min0, min1, ..., minn]}
    """
    norm_param = {
        'type': 'zero-one',
        'max': [],
        'min': []
    }
    L = len(data_set[0])
    data_normalized = []

    for l in range(L):
        data = data_set[:, l]
        data_max = np.max(data[begin:end])
        data_min = np.min(data[begin:end])
        norm_param['max'].append(data_max)
        norm_param['min'].append(data_min)
        data = (data - data_min) / (data_max - data_min)
        data_normalized.append(data)

    return data_normalized, norm_param



def denormalization(data_normalized, norm_param):
    """
    数据集反归一化,零一归一化并返回归一化参数
    Input输入:
    data_normalized:nparray,输入需要反归一化的径流数据。
    norm_param:dict,归一化参数列表。  norm_param 为{'type': 'zero-one', 'max':[max0, max1, ..., maxn],'min':[min0, min1, ..., minn]}
                                                    max0,min为反归一化所需要参数
    Output输出:
    denormalization:nparray,输出反归一化对应的径流预测值。

    """
    if norm_param['type'] == 'zero-one':
        data_max = norm_param['max'][0]
        data_min = norm_param['min'][0]
        data_denormalization = np.multiply(data_normalized, (data_max - data_min)) + data_min

    return data_denormalization
    

def missing_value_process(data):
    # 线性插值
    if np.min(data) == -999:
        position = np.argwhere(data == -999)[:, 0]
        i = 0
        while i < len(position):
            begin = position[i]
            end = position[i]
            while(end + 1 in position):
                i = i + 1
                end = position[i]
            linedata = np.linspace(data[begin-1],data[end+1], end-begin+1+2)
            data[begin:end+1, :] = linedata[1:-1, :]
            i = i + 1
        return data
    else:
        return data
        


def process(train_size, useday_n, predict_day, selected_id, start_date, end_date, streamflow_path, forcing_data_path):
    qobs = xr.open_dataset(streamflow_path)
    mfobs = xr.open_dataset(forcing_data_path)
    # data_Time = np.datetime_as_string(qobs.time.to_pandas(), unit='D').reshape(-1, 1)
    data_Discharge = np.array(qobs.sel(station_id=selected_id, time=slice(start_date, end_date)).discharge.to_pandas()).reshape(-1, 1)
    data_Discharge = missing_value_process(data_Discharge)
    data_Prcp = np.array(mfobs.sel(station_id=selected_id, time=slice(start_date, end_date)).prcp.to_pandas()).reshape(-1, 1)
    data_Tmax = np.array(mfobs.sel(station_id=selected_id, time=slice(start_date, end_date)).tmax.to_pandas()).reshape(-1, 1)
    data_Tmin = np.array(mfobs.sel(station_id=selected_id, time=slice(start_date, end_date)).tmin.to_pandas()).reshape(-1, 1)

    data_set = np.concatenate([data_Discharge, data_Prcp, data_Tmax, data_Tmin], axis=1)

    '''
    time_list = [i[0] for i in data_Time]
    df = pd.DataFrame(data_set,
                      index=time_list,
                      columns=['discharge', 'prcp', 'tmax', 'tmin'])
    '''
    #分割数据集
    total_num = int(len(data_Discharge)) - useday_n - (predict_day - 1)
    train_num = int(total_num * train_size)
    
    # 归一化
    data_normalized, norm_param= nomarlization_01(data_set=data_set, begin=0, end=train_num + useday_n)
    x_train, y_train = dataset_split(dataset=data_normalized,
                                     begin=0,
                                     end=train_num + useday_n,
                                     useday_n=useday_n,
                                     predict_day=predict_day)
    x_test, y_test = dataset_split(dataset=data_normalized,
                                   begin=train_num,
                                   end=total_num + useday_n,
                                   useday_n=useday_n,
                                   predict_day=predict_day)

    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return x_train, y_train, x_test, y_test, norm_param
    """    
    train_data_set = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
    df_train = pd.DataFrame(train_data_set)
    df_train.to_csv(train_set_path, index=False)
    test_data_set = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)
    df_test = pd.DataFrame(test_data_set)
    df_test.to_csv(test_set_path, index=False) 
    """
    
    

    
    



