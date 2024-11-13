import pandas as pd
import matplotlib.pyplot as plt
import shap

# 特徵重要性分析
def feature_importance_analysis(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")

# 列出前20重要特徵
def get_top_20_features(model, X):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
    print(top_features)
    top_features.plot(kind='bar', x='feature', y='importance', legend=False, title="Top 20 Features")
    plt.show()
