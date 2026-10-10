import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit

"""
    定义所有基模型网格搜索的超参数
"""
class parameters():
    param_grid_svr = {
        'C': np.logspace(-5, 2, 10),
        'gamma': np.logspace(-2, 2, 10),
        'epsilon': [0.001, 0.01]
    }
    param_grid_radomforest = {
        'n_estimators':[100, 200, 500],
        'min_samples_split':[2, 5, 10],
        'max_depth':[3, 5, 7, 10],
        'min_samples_leaf':[1, 2, 5]
    }
    param_grid_bp = {
        'hidden_layer_sizes':
        [(i, j) for i in range(10, 110, 10)
        for j in range(10, 110, 10)] + [(i, ) for i in range(10, 110, 10)],
        'alpha': [0.0001, 0.001, 0.01]
    }



def search_bestmodel(model, x_train, y_train, param_grid_name:str, station_id:str, filepath:str='.'):
    """
    search_bestmodel使用4折的时序序列划分对基模型进行网格搜索
        Input输入:
            model:需要训练的模型。
            param_grid_name:str,需要训练的模型的名称（svr/forest/BP）。
            x_train, y_train:用于训练的训练集特征和标签。
            station_id:流域编号。
            filepath:交叉验证结果csv的输出目录。
        Output输出:
            best_estimator_:通过网格搜索所得到的最优模型。
            同时在 filepath/cv_result/ 下存储排名前10的参数组合及得分的csv文件。 
    """
    cv_split = TimeSeriesSplit(n_splits=4, test_size=int(len(x_train)*0.2))
    if param_grid_name == "svr":
        param = parameters.param_grid_svr
    elif param_grid_name == "forest": 
        param = parameters.param_grid_radomforest
    elif param_grid_name == "BP":
        param = parameters.param_grid_bp
    column = []
    for c in param.keys():
        column.append("param_"+c)
    column = column + ['mean_train_score','mean_test_score','rank_test_score']
    grid_search = GridSearchCV(estimator=model, param_grid= param, cv= cv_split, return_train_score=True, n_jobs=10, verbose=3)
    grid_search.fit(x_train, y_train.ravel())
    Re = pd.DataFrame(grid_search.cv_results_)
    f = Re[column].sort_values(by='rank_test_score').head(10)
    os.makedirs(filepath + "/cv_result", exist_ok=True)  # 目录不存在时自动创建
    f.to_csv(filepath + "/cv_result/cv_" + station_id + "_" + param_grid_name + ".csv",index=True,encoding='utf-8')
#    保存最优模型
#    pkl_filename = "./cv_result/cv_" + station_id + "_" + param_grid_name + ".pkl"
#    with open(pkl_filename, 'wb') as file:
#        pickle.dump(grid_search.best_estimator_, file) 
#    file.close()
    return grid_search.best_estimator_

