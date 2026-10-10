import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
'''
    计算μ
    Input输入：
    Models：List，输入所有的基学习器，格式为list[('Model_name',Model),…]。
    x：numpy,训练集数据。
    y：numpy，训练集输出，经过Cox-Box变换。
    f: List，各个基学习器对各个数据的预测结果。
    lam，C：cox-box变换参数。
    Output输出： 
    f：List，各个基学习器对各个数据的预测结果。
'''

def mu(Models, x, lam, C):
    num_datasize = len(x)
    num_models = len(Models)
    f = []
    for k in range(num_models):
        temp = Models[k][1].predict(x)
        for t in range(len(temp)):
            if temp[t] <= -C:
                temp[t] = -0.999 * C  #防止报错
        f.append(stats.boxcox(temp + C, lam))
    return f


'''
    计算方差σ²
    Input输入：
    Models：List，输入所有的基学习器，格式为list[('Model_name',Model),…]。
    y：numpy，训练集输出，经过Cox-Box变换。
    f: List，各个基学习器对各个数据的预测结果。
    Output输出：
    V：list,每一个基学习器对饮的方差[σ1²,σ2²,...,σk²]。
'''

def var0(f, y):
    num_models = len(f)
    y = np.array(y)
    temp = []
    V = 0.0
    for k in range(num_models):
        temp.append((np.subtract(f[k], y))**2)
        V += temp[k].mean()
    V = V / num_models
    V_list = [V for i in range(num_models)]
    return V_list


def var(z, f, y):
    num_models = len(f)
    y = np.array(y)
    V = []
    z = np.array(z)
    for k in range(num_models):
        Z = z[:, k]
        V.append(((np.subtract(f[k], y))**2 * Z).sum() / Z.sum())
    return V


'''
    计算g(Q|f,σ²)条件概率
    Input输入：
    fs：List，所有的基学习器的预测结果。
    var: List，各个基学习器对应的方差。
    Output输出：
    G：List，基学习器的隐变量。格式为[array([g11,g21,...,zt1]),
                                              ...,
                                      array([z1k,z2k,...,ztk])]
'''

import scipy.stats as stats
from math import sqrt

def g_cal(fs, var, Q):
    g = []
    for k in range(len(fs)):
        if var[k]==0: #若为偏差极大的数则防止报错赋给一个很小的值
            var[k]= 10**(-38)
        g.append(stats.norm.pdf(Q, loc=fs[k], scale=sqrt(var[k])))
    return g


'''
    likelihood计算似然值
    Input输入：
    g：List，模型条件概率值。
    w：List，模型权重。
    Output输出：
    likelihood :float，极大似然值。
'''
from math import log


def likelihood(g, w):
    total_likelihood = 0
    for t in range(len(g[0])):
        temp = 0
        for k in range(len(w)):
            temp += w[k] * g[k][t]
        if temp == 0:  #下限溢出
            temp = 10**(-38)
        total_likelihood += log(temp)
    return total_likelihood



'''
    z_list计算模型的隐变量权重
    输入：
    g：List，条件概率值。
    Output输出：
    z :List，基学习器的隐变量。格式为[[z11,z12,...,z1k],
                                              ...,
                                      [zt1,zt2,...,ztk]]
'''

def z_list(g):
    k_len = len(g)
    zero_situation = [1 / k_len for n in range(k_len)]
    Z = []
    for t in range(len(g[0])):
        temp = 0.0
        temp_list = []
        for k in range(k_len):
            temp += g[k][t]
            temp_list.append(g[k][t])
        if temp == 0:
            Z.append(zero_situation)
        else:
            Z.append([temp_list[i] / temp for i in range(k_len)])
    return Z


'''
    W_cal计算模型的变量权重
    输入：
    z :List，基学习器的隐变量。
    Output输出：
    W :List，基学习器的更新后的权重。
'''

def W_cal(z_list):
    k_len = len(z_list[0])
    W = [0.0 for i in range(k_len)]
    for l in z_list:
        W = [W[k] + l[k] for k in range(k_len)]
    W = [w / len(z_list) for w in W]
    return W

'''
    Input输入：
    Models：List，输入所有的基学习器，格式为list[('Model_name',Model),…]。
    Deta：float，M步计算所得的似然值likelihood与上一步迭代的差值是否小于Deta，是则停止迭代，反之继续。
    Output输出：
    ω :List，各个基学习器的权重，其和为1。
'''


def BMA_Weight(Models, Deta, x_train, y_train, lam_best, C, MaxIter=10):
    Iter = 0  # 迭代次数
    l_Deta = float('inf')
    W = [[1 / len(Models) for n in range(len(Models))]]
    L = []  # 极大似然值
    f = mu(Models=Models, x=x_train, lam=lam_best, C=C)  # 期望
    Vars = []  # 方差
    
    y_BOXCOX = stats.boxcox(y_train + C, lam_best)
    Vars.append(var0(f=f, y=y_BOXCOX))

    while (l_Deta > Deta) and (Iter < MaxIter):
        g = g_cal(fs=f, var=Vars[Iter], Q=y_BOXCOX)
        L.append(likelihood(g=g, w=W[Iter]))
        if Iter > 0:
            l_Deta = abs(L[Iter] - L[Iter - 1])
        Z = z_list(g=g)  # 隐藏变量
        print('第%d次迭代，极大似然差值为%.5f' % (Iter, l_Deta))
        print('权重为：')
        print(W[Iter])
        Iter += 1

        W.append(W_cal(Z))
        Vars.append(var(z=Z, f=f, y=y_train))
    return W[Iter]

def BMA_Model_predict(Models, Model_Ws, x_test):
    predictions = [0 for i in range(len(x_test))]
    k = 0
    for Model in Models:
        predictions = predictions + Model[1].predict(x_test) * Model_Ws[k]
        k = k + 1
    return predictions

def predict_with_error_BMA(Models, Model_Ws, use_day, predict_day, x_train, x_test):
    max_predict_day = max(predict_day)
    test_len = len(x_test)
    new_X = np.concatenate((x_train[-(max_predict_day + use_day - 1):], x_test), axis=0)
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
        pre_n[n] = BMA_Model_predict(Models, Model_Ws, X)
    return_list = []
    for need in predict_day:
        return_list.append(pre_n[need-1][-test_len:])
    return return_list
