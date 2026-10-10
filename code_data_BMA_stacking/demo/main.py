import pandas as pd
import numpy as np

import os

import sys
sys.path.append('..')

from models.BMA import predict_with_error_BMA, BMA_Weight, BMA_Model_predict
from models.Stacking import Stacking
from util.data_process import process, denormalization
from util.estimators import all_estimators, evalue
from util.BOX_COX import BOX_COX
from util.grid_search import search_bestmodel
from util.predict import predict_with_error

class config():
    streamflow_path = '../data/camels_streamflow.nc'
    forcing_data_path = '../data/camels_forcingmete.nc'
    # 选定前n天的驱动因素为特征
    n = 7
    # 预测天数
    predict_day = [1, 2, 3, 4, 5, 6, 7]
    outputpath = '../outputs'
    #  选取开始时间
    start_date = '1995-01-01'
    #  选取结束时间
    end_date = '2004-12-31'

gauge = pd.read_csv("../camels_name.txt", header=0, sep=";", dtype=str)

# 首次运行时自动创建输出目录
for out_dir in [config.outputpath + "_BMA", config.outputpath + "_Stacking", config.outputpath + "/cv_result"]:
    os.makedirs(out_dir, exist_ok=True)

processed_id = []  # 用于存储.txt文件的文件名

for filename in os.listdir(config.outputpath+"_BMA"):
    if filename.endswith('.csv'):
        processed_id.append(filename[:-4].split('_')[-1])  # gauge名称 
unprocessed_id = []
for id in gauge["gauge_id"]:
    if (id not in processed_id):
        unprocessed_id.append(id)

# unprocessed_id = reversed(unprocessed_id)

for id in unprocessed_id:
    if id == '03281100':
        continue
    x_train, y_train, x_test, y_test, norm_param = process(train_size=0.8, useday_n=config.n, selected_id=id, start_date=config.start_date, end_date=config.end_date,
                                                           predict_day=1, streamflow_path=config.streamflow_path, forcing_data_path=config.forcing_data_path)
    myestimators = []
    # 模型搜索
    Model_evaluation = []
    for n in range(len(all_estimators)):
        myestimators.append((all_estimators[n][0], search_bestmodel(all_estimators[n][1], x_train, y_train, all_estimators[n][0], id)))
        row = [myestimators[n][0]]
        train_score = evalue(y_true=denormalization(y_train, norm_param), y_pred=denormalization(myestimators[n][1].predict(x_train),norm_param))
        for k in train_score.keys():
            row.append(train_score[k])
        test_predictions = predict_with_error(model=myestimators[n][1], use_day=config.n, predict_day=config.predict_day,
                                              x_train=x_train, x_test=x_test)
        for t in range(len(config.predict_day)):
            day = config.predict_day[t]
            test_score = evalue(y_true=denormalization(y_test, norm_param), y_pred=denormalization(test_predictions[t],norm_param))
            for k in test_score.keys():
                row.append(test_score[k])
        Model_evaluation.append(row)

    columns=['模型类型']
    for k in train_score.keys():
        columns.append('train' + '_' + k)
    for t in range(len(config.predict_day)):
        for k in test_score.keys():          
            columns.append('test' + '_' + k + '_' + str(config.predict_day[t]))
    
    df = pd.DataFrame(data=Model_evaluation, columns=columns)
    df.to_csv('../outputs/cv_result/' + id + '.csv', encoding='utf_8_sig')

    # 使数据服从正态分布(box-cox变换)
    moudle_outdata = y_train
    for estimator in myestimators:
        moudle_outdata = np.concatenate((moudle_outdata, estimator[1].predict(x_train)), axis=0)
    
    # box-cox变换 
    bC, bl = BOX_COX(moudle_outdata)

    # BMA 使用 svr + forest + BP 全部三个模型这一种组合
    Model_evaluation = []
    Ms = myestimators
    modelname = []
    for m in Ms:
        modelname.append(m[0])

    W_BMA = BMA_Weight(Ms, 0.001, x_train, y_train, lam_best=bl, C=bC, MaxIter=100)
    row = [modelname, W_BMA]

    train_predictions = BMA_Model_predict(Models=Ms, Model_Ws=W_BMA, x_test=x_train)
    train_score = evalue(y_true=denormalization(y_train, norm_param), y_pred=denormalization(train_predictions,norm_param))
    for k in train_score.keys():
        row.append(train_score[k])
    test_predictions = predict_with_error_BMA(Models=Ms, Model_Ws = W_BMA, use_day=config.n, predict_day=config.predict_day,
                                              x_train=x_train, x_test=x_test)

    for t in range(len(config.predict_day)):
        day = config.predict_day[t]
        test_score = evalue(y_true=denormalization(y_test, norm_param), y_pred=denormalization(test_predictions[t], norm_param))
        for k in test_score.keys():
            row.append(test_score[k])
    Model_evaluation.append(row)

    columns=['模型组合','对应权重']
    for k in train_score.keys():
        columns.append('train' + '_' + k)
    for t in range(len(config.predict_day)):
        for k in test_score.keys():          
            columns.append('test' + '_' + k + '_' + str(config.predict_day[t]))
    df = pd.DataFrame(data=Model_evaluation, columns=columns)
    df.to_csv(config.outputpath + "_BMA" + '/outputs_BMA_' + id + '.csv', encoding='utf_8_sig')

    # Stacking 使用 svr + forest + BP 这一种组合，第二层用Ridge回归
    Model_evaluation_Ridge = []
    Ms = myestimators
    modelname = []
    for m in Ms:
        modelname.append(m[0])

    # Ridge回归做第二层
    stacking_Ridge = Stacking(Ms, "Ridge", x_train, y_train)

    row = ["Stacking_Ridge", modelname]
    train_predictions_Ridge = stacking_Ridge.predict(x_train)
    train_score_Ridge = evalue(y_true=denormalization(y_train, norm_param), y_pred=denormalization(train_predictions_Ridge, norm_param))
    for k in train_score_Ridge.keys():
        row.append(train_score_Ridge[k])
    test_predictions = predict_with_error(model=stacking_Ridge, use_day=config.n, predict_day=config.predict_day,
                                                  x_train=x_train, x_test=x_test)   

    for t in range(len(config.predict_day)):
        day = config.predict_day[t]
        test_score_Ridge = evalue(y_true=denormalization(y_test, norm_param), y_pred=denormalization(test_predictions[t], norm_param))
        for k in test_score_Ridge.keys():
            row.append(test_score_Ridge[k])
    Model_evaluation_Ridge.append(row)  # 每个流域只写一行结果

    columns=['模型类型', '组合模型']
    for k in train_score.keys():
        columns.append('train' + '_' + k)
    for t in range(len(config.predict_day)):
        for k in test_score.keys():
            columns.append('test' + '_' + k + '_' + str(config.predict_day[t]))

    df = pd.DataFrame(data=Model_evaluation_Ridge, columns=columns)
    df.to_csv(config.outputpath + "_Stacking" + '/outputs_Stacking_Ridge_' + id + '.csv', encoding='utf_8_sig')
