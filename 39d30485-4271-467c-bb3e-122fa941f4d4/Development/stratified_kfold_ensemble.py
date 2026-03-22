from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
import scipy.special as sp
import numpy as np

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_final = np.zeros(len(X_clean))
cat_final_oof = np.zeros(len(X_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean, y)):
    print(f"Fold {fold+1}", flush=True)

    X_tr = X_clean.iloc[train_idx]
    X_vl = X_clean.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_vl = y.iloc[val_idx]

    lgb = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
            eval_metric='average_precision',
            callbacks=[early_stopping(50, verbose=False)])
    lgb_preds = lgb.predict_proba(X_vl)[:, 1]
    oof_final[val_idx] = lgb_preds

    cat = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, random_seed=42, eval_metric='PRAUC',
        verbose=0, early_stopping_rounds=50
    )
    cat.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
    cat_preds = cat.predict_proba(X_vl)[:, 1]
    cat_final_oof[val_idx] = cat_preds

    lgb_score = average_precision_score(y_vl, lgb_preds)
    cat_score = average_precision_score(y_vl, cat_preds)
    print(f"  LGB: {lgb_score:.5f} | CAT: {cat_score:.5f}")

lgb_logit = sp.logit(oof_final.clip(1e-6, 1-1e-6))
cat_logit = sp.logit(cat_final_oof.clip(1e-6, 1-1e-6))
ensemble = sp.expit(0.3 * lgb_logit + 0.7 * cat_logit)

print(f"\nLGB      : {average_precision_score(y, oof_final):.5f}")
print(f"CAT      : {average_precision_score(y, cat_final_oof):.5f}")
print(f"ENSEMBLE : {average_precision_score(y, ensemble):.5f}")
print(f"PREVIOUS : 0.06621")