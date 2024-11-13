import pandas as pd

# 加載數據
def load_data(file_path):
    return pd.read_csv(file_path)

# 保存數據
def save_data(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
