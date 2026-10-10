from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
import numpy as np

def Stacking(Models, final_estimator_name, x_train, y_train):
    """
    Stacking 对输入的基模型进行二层堆叠集成
        Input输入:
            Models:list,所有基学习器的列表,list[('Model_name', Model),…]。
            final_estimator_name:str,第二层（元）学习器的名称,目前只使用"Ridge"。
            x_train, y_train:用于训练的训练集特征和标签。
        Output输出:
            stacking_reg:拟合完成的 StackingRegressor 模型（元学习器为 RidgeCV）。
    """
    if final_estimator_name == "Ridge":
        stacking_reg = StackingRegressor(estimators=list(Models),final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10),
                                        fit_intercept=True, scoring='r2', gcv_mode=None,
                                        store_cv_results=False),n_jobs=-1)
    else:
        raise ValueError("仅支持Ridge作为第二层学习器")

    stacking_reg.fit(X=x_train, y=y_train)
    return stacking_reg
