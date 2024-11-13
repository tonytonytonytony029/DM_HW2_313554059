import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 加載數據
def load_data(file_path):
    return pd.read_csv(file_path)

# 數據清理與處理
def preprocess_data(df):
    # 處理缺失值
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # 標準化數值型特徵
    scaler = StandardScaler()
    numeric_features = df.select_dtypes(include=['float64', 'int']).columns
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # 處理類別型特徵
    encoder = OneHotEncoder(drop='first', sparse=False)
    categorical_features = df.select_dtypes(include=['object']).columns
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
    df = df.drop(categorical_features, axis=1).join(encoded_df)
    
    return df

# 處理數據不平衡
def balance_data(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# 可視化數據分佈
def visualize_data(df):
    for col in df.select_dtypes(include=['float64', 'int']):
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.show()

# 主函數：加載與處理數據
def prepare_data(file_path):
    df = load_data(file_path)
    visualize_data(df)
    X, y = df.drop('has_died', axis=1), df['has_died']
    X = preprocess_data(X)
    X, y = balance_data(X, y)
    return X, y
