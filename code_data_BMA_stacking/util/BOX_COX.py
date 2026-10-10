from scipy import stats
import numpy as np

def BOX_COX(data):
    """
    数据集分割 BOX_COX,对所有径流数据,求BOX-COX变换的最优参数
        Input输入:
            data:nparray,输入所有的数据。
        Output输出:
            best_C, best_l:float,输入对应的BOX-COX变换的平移C和参数lam。
    """
    M = min(data)
    if M > 0:
        R = np.linspace(0, 0.1, num=500)  #0.005是起点，0.01是终点，num是生成列表元素的数目
    else:
        R = np.linspace(-M + 0.0001, -M + 0.1, num=500)  #0.005是起点，0.01是终点，num是生成列表元素的数目
    Max_p = 0.0
    best_C = 0.0
    best_l = 0.0
    for C in R:
        y_boxcox, lmbda = stats.boxcox(data + C)
        # 使用 normaltest 测试检验正态性
        _, normaltest_p_value = stats.normaltest(y_boxcox)
        if normaltest_p_value >= Max_p:
            best_C = C
            Max_p = normaltest_p_value
            best_l = lmbda
    return best_C, best_l