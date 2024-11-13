 # data_imputation.py
"""
缺失值補全腳本：
包含不同的缺失值插補方法。
"""

import pandas as pd
from sklearn.impute import KNNImputer

def mean_impute(data, column):
    """使用平均值插補缺失值"""
    data[column].fillna(data[column].mean(), inplace=True)
    return data

def knn_impute(data, columns, n_neighbors=5):
    """使用 KNN 插補缺失值"""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    data[columns] = imputer.fit_transform(data[columns])
    return data

# 加載數據，進行插補
train_data = pd.read_csv("train_X.csv")
train_data = mean_impute(train_data, 'column_name')
train_data = knn_impute(train_data, ['column1', 'column2'])

