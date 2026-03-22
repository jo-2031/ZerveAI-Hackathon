from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import numpy as np

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cat_oof2 = np.zeros(len(X_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean, y)):
    print(f"Fold {fold+1}", flush=True)

    X_tr = X_clean.iloc[train_idx]
    X_vl = X_clean.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_vl = y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='PRAUC',
        verbose=0,
        early_stopping_rounds=100
    )

    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))

    preds = model.predict_proba(X_vl)[:, 1]
    cat_oof2[val_idx] = preds
    print(f"  PR-AUC: {average_precision_score(y_vl, preds):.5f}")

print(f"\nSTRONGER CATBOOST: {average_precision_score(y, cat_oof2):.5f}")