
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

# KFold交叉驗證
def kfold_validation(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    f1_scorer = make_scorer(f1_score, average='macro')
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

    f1_scores = cross_val_score(model, X, y, cv=skf, scoring=f1_scorer)
    auc_scores = cross_val_score(model, X, y, cv=skf, scoring=auc_scorer)
    
    print(f"Average F1-Score: {f1_scores.mean():.4f}")
    print(f"Average AUROC: {auc_scores.mean():.4f}")
    return f1_scores.mean(), auc_scores.mean()
