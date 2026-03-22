from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import numpy as np

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cat_oof = np.zeros(len(X_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean, y)):
    print(f"Fold {fold+1}", flush=True)

    X_tr = X_clean.iloc[train_idx]
    X_vl = X_clean.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_vl = y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='PRAUC',
        verbose=0,
        early_stopping_rounds=50
    )

    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))

    preds = model.predict_proba(X_vl)[:, 1]
    cat_oof[val_idx] = preds
    print(f"  PR-AUC: {average_precision_score(y_vl, preds):.5f}")

print(f"\nCATBOOST PR-AUC: {average_precision_score(y, cat_oof):.5f}")

# Ensemble immediately
import scipy.special as sp
lgb_logit = sp.logit(oof_preds.clip(1e-6, 1-1e-6))
cat_logit = sp.logit(cat_oof.clip(1e-6, 1-1e-6))
final_preds = sp.expit(0.5 * lgb_logit + 0.5 * cat_logit)
print(f"ENSEMBLE PR-AUC: {average_precision_score(y, final_preds):.5f}")