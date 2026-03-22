from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import numpy as np

# Scale features first (required for MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
mlp_oof = np.zeros(len(X_clean))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
    print(f"Fold {fold+1}", flush=True)

    X_tr = X_scaled[train_idx]
    X_vl = X_scaled[val_idx]
    y_tr = y.iloc[train_idx]
    y_vl = y.iloc[val_idx]

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    model.fit(X_tr, y_tr)

    preds = model.predict_proba(X_vl)[:, 1]
    mlp_oof[val_idx] = preds
    print(f"  PR-AUC: {average_precision_score(y_vl, preds):.5f}")

print(f"\nMLP PR-AUC: {average_precision_score(y, mlp_oof):.5f}")

# Ensemble all three
import scipy.special as sp
lgb_logit = sp.logit(oof_preds.clip(1e-6, 1-1e-6))
cat_logit = sp.logit(cat_oof.clip(1e-6, 1-1e-6))
mlp_logit = sp.logit(mlp_oof.clip(1e-6, 1-1e-6))
final_preds = sp.expit((lgb_logit + cat_logit + mlp_logit) / 3)
print(f"3-MODEL ENSEMBLE PR-AUC: {average_precision_score(y, final_preds):.5f}")