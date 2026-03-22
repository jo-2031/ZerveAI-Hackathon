from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier, early_stopping
import numpy as np

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean, y)):
    print(f"Fold {fold+1}", flush=True)

    X_tr = X_clean.iloc[train_idx]
    X_vl = X_clean.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_vl = y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        eval_metric='average_precision',
        callbacks=[early_stopping(50, verbose=False)]
    )

    preds = model.predict_proba(X_vl)[:, 1]
    oof_preds[val_idx] = preds
    print(f"  PR-AUC: {average_precision_score(y_vl, preds):.5f}")

print(f"\nFINAL PR-AUC: {average_precision_score(y, oof_preds):.5f}")