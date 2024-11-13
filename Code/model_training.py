from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import joblib

# 初始化模型
def initialize_models():
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

# 模型訓練
def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

# 評估模型
def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_proba)
    return f1, auc

# 保存模型
def save_model(model, model_name='model.joblib'):
    joblib.dump(model, model_name)
    print(f"Model saved as {model_name}")
