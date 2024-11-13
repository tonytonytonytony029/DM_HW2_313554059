# data_analysis.py
"""
擴展的數據集分析腳本：
進行詳細的資料探索，包括統計摘要、分佈分析、相關性分析、缺失值檢查等。
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """讀取數據集"""
    return pd.read_csv(file_path)

def basic_info(data):
    """檢查基本資訊"""
    print("Data Shape:", data.shape)
    print("Data Types:\n", data.dtypes)

def missing_values_analysis(data):
    """缺失值分析"""
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data)) * 100
    print("Missing Values:\n", missing_data[missing_data > 0])
    print("Missing Percentage:\n", missing_percentage[missing_percentage > 0])

def describe_data(data):
    """打印數據的統計摘要"""
    print(data.describe())

def visualize_distribution(data, column):
    """可視化單個特徵的分佈"""
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_correlation_matrix(data):
    """繪製相關矩陣熱圖"""
    plt.figure(figsize=(10, 8))
    correlation = data.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def category_distribution(data, column):
    """類別特徵的分佈檢查"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=data)
    plt.title(f'Category Distribution of {column}')
    plt.show()

def target_distribution(target):
    """目標變量分佈檢查"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target)
    plt.title('Target Variable Distribution')
    plt.show()

# 示例
train_data = load_data("train_X.csv")
train_target = load_data("train_y.csv")

# 基本資訊
basic_info(train_data)

# 缺失值分析
missing_values_analysis(train_data)

# 統計摘要
describe_data(train_data)

# 分佈檢查
for column in train_data.select_dtypes(include=['float64', 'int64']).columns:
    visualize_distribution(train_data, column)

# 類別特徵分佈檢查
for column in train_data.select_dtypes(include=['object']).columns:
    category_distribution(train_data, column)

# 目標變量分佈
target_distribution(train_target)
