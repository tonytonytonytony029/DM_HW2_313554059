from sklearn.metrics import f1_score, roc_auc_score

# 計算F1-Score
def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

# 計算AUROC
def calculate_auc(y_true, y_proba):
    return roc_auc_score(y_true, y_proba[:, 1])
