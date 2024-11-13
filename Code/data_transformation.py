 # data_transformation.py
"""
數據轉換腳本：
執行編碼和標準化，轉換數據為模型適用格式。
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def encode_categorical(data, column):
    """對類別特徵進行 One-Hot 編碼"""
    encoder = OneHotEncoder(sparse=False)
    transformed = encoder.fit_transform(data[[column]])
    encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out([column]))
    data = data.drop(column, axis=1).join(encoded_df)
    return data

def standardize_data(data, columns):
    """對數值特徵進行標準化"""
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# 加載數據，進行轉換
train_data = pd.read_csv("train_X.csv")
train_data = encode_categorical(train_data, 'categorical_column')
train_data = standardize_data(train_data, ['numeric_column1', 'numeric_column2'])

