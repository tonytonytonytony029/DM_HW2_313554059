 # data_cleaning.py
"""
數據清理腳本：
包括處理缺失值和異常值的函數，提升數據品質。
"""

import pandas as pd

def drop_missing_values(data, threshold=0.5):
    """刪除缺失值比例超過閾值的列"""
    data = data.loc[:, data.isnull().mean() < threshold]
    return data

def fill_missing_values(data, column, method='mean'):
    """填充缺失值：支持 'mean', 'median' 或 'mode'"""
    if method == 'mean':
        data[column].fillna(data[column].mean(), inplace=True)
    elif method == 'median':
        data[column].fillna(data[column].median(), inplace=True)
    elif method == 'mode':
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data

# 加載數據並進行清理
train_data = pd.read_csv("train_X.csv")
train_data = drop_missing_values(train_data)
train_data = fill_missing_values(train_data, 'column_name', method='mean')

