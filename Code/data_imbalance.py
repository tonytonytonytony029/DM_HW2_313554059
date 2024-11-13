 # data_imbalance.py
"""
處理數據不平衡腳本：
使用 SMOTE 等技術平衡目標變量。
"""

import pandas as pd
from imblearn.over_sampling import SMOTE

def balance_data(data, target_column):
    """使用 SMOTE 進行欠採樣"""
    smote = SMOTE()
    X, y = data.drop(columns=[target_column]), data[target_column]
    X_res, y_res = smote.fit_resample(X, y)
    return pd.concat([X_res, y_res], axis=1)

# 加載數據，進行不平衡處理
train_X = pd.read_csv("train_X.csv")
train_y = pd.read_csv("train_y.csv")
train_data = pd.concat([train_X, train_y], axis=1)
balanced_data = balance_data(train_data, 'has_died')
