import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import scipy.special as sp

# Step 1: Load test file
test = pd.read_csv("test_data_hackathon.csv", low_memory=False)
test = test.replace("", np.nan)
print("Test shape:", test.shape)
print("Test columns:", test.columns.tolist()[:5])

# Step 2: Prepare test features
X_test = test.drop(columns=['id']).copy()

# Step 3: Missing handling (use TRAIN medians)
missing_cols = ['feature_8', 'feature_9', 'feature_12', 'feature_15',
                'feature_28', 'feature_29', 'feature_31', 'feature_34',
                'feature_35', 'feature_38', 'feature_39', 'feature_42', 'feature_45']

for col in missing_cols:
    X_test[col + '_missing'] = X_test[col].isnull().astype(int)

X_test['feature_8'] = X_test['feature_8'].fillna(-999)
for col in missing_cols:
    if col != 'feature_8':
        median_val = X_clean[col].median()
        X_test[col] = X_test[col].fillna(median_val)

# Step 4: Feature engineering
X_test['f22_x_f16'] = X_test['feature_22'] * X_test['feature_16']
X_test['f22_x_f10'] = X_test['feature_22'] * X_test['feature_10']
X_test['f16_x_f10'] = X_test['feature_16'] * X_test['feature_10']
X_test['f22_minus_f30'] = X_test['feature_22'] - X_test['feature_30']
X_test['f16_minus_f30'] = X_test['feature_16'] - X_test['feature_30']
X_test['f22_plus_f16'] = X_test['feature_22'] + X_test['feature_16']
X_test['f22_div_f30'] = X_test['feature_22'] / (X_test['feature_30'] + 0.001)
X_test['f10_div_f30'] = X_test['feature_10'] / (X_test['feature_30'] + 0.001)
X_test['f11_x_f22'] = X_test['feature_11'] * X_test['feature_22']
X_test['f11_x_f21'] = X_test['feature_11'] * X_test['feature_21']
X_test['f22_x_f21'] = X_test['feature_22'] * X_test['feature_21']
X_test['f11_x_f16'] = X_test['feature_11'] * X_test['feature_16']
X_test['risk_score'] = (X_test['feature_11'] + X_test['feature_22'] +
                        X_test['feature_21'] + X_test['feature_16'] -
                        X_test['feature_20'] - X_test['feature_30'])
X_test['f24_x_f22'] = X_test['feature_24'] * X_test['feature_22']
X_test['f24_x_f11'] = X_test['feature_24'] * X_test['feature_11']
X_test['f10_x_f22'] = X_test['feature_10'] * X_test['feature_22']
X_test['f10_x_f11'] = X_test['feature_10'] * X_test['feature_11']

print("X_test shape:", X_test.shape)
print("X_clean shape:", X_clean.shape)
print("Columns match:", X_test.shape[1] == X_clean.shape[1])

# Step 5: Train on FULL training data
print("\nTraining LightGBM on full data...")
lgb_final = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1
)
lgb_final.fit(X_clean, y)
lgb_test_preds = lgb_final.predict_proba(X_test)[:, 1]
print("LightGBM done")

print("Training CatBoost on full data...")
cat_final = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, random_seed=42, verbose=0
)
cat_final.fit(X_clean, y)
cat_test_preds = cat_final.predict_proba(X_test)[:, 1]
print("CatBoost done")

# Step 6: Ensemble with best weights
lgb_logit = sp.logit(lgb_test_preds.clip(1e-6, 1-1e-6))
cat_logit = sp.logit(cat_test_preds.clip(1e-6, 1-1e-6))
final_test_preds = sp.expit(0.3 * lgb_logit + 0.7 * cat_logit)

# Step 7: Save submission
submission = pd.DataFrame({
    'id': test['id'],
    'target': final_test_preds
})
submission.to_csv('submission.csv', index=False)

print("\nSubmission saved!")
print("Shape:", submission.shape)
print(submission.head(10))