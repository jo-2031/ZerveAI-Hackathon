# Keep existing engineered features
X_clean['f22_x_f16'] = X_clean['feature_22'] * X_clean['feature_16']
X_clean['f22_x_f10'] = X_clean['feature_22'] * X_clean['feature_10']
X_clean['f16_x_f10'] = X_clean['feature_16'] * X_clean['feature_10']
X_clean['f22_minus_f30'] = X_clean['feature_22'] - X_clean['feature_30']
X_clean['f16_minus_f30'] = X_clean['feature_16'] - X_clean['feature_30']
X_clean['f22_plus_f16'] = X_clean['feature_22'] + X_clean['feature_16']
X_clean['f22_div_f30'] = X_clean['feature_22'] / (X_clean['feature_30'] + 0.001)
X_clean['f10_div_f30'] = X_clean['feature_10'] / (X_clean['feature_30'] + 0.001)

# NEW — combinations of strongest binary separators
X_clean['f11_x_f22'] = X_clean['feature_11'] * X_clean['feature_22']
X_clean['f11_x_f21'] = X_clean['feature_11'] * X_clean['feature_21']
X_clean['f22_x_f21'] = X_clean['feature_22'] * X_clean['feature_21']
X_clean['f11_x_f16'] = X_clean['feature_11'] * X_clean['feature_16']
X_clean['risk_score'] = (X_clean['feature_11'] + X_clean['feature_22'] + 
                         X_clean['feature_21'] + X_clean['feature_16'] - 
                         X_clean['feature_20'] - X_clean['feature_30'])
X_clean['f24_x_f22'] = X_clean['feature_24'] * X_clean['feature_22']
X_clean['f24_x_f11'] = X_clean['feature_24'] * X_clean['feature_11']
X_clean['f10_x_f22'] = X_clean['feature_10'] * X_clean['feature_22']
X_clean['f10_x_f11'] = X_clean['feature_10'] * X_clean['feature_11']

print("Shape:", X_clean.shape)

# Verify risk_score separates classes
pos_risk = df[df['target']==1].index
neg_risk = df[df['target']==0].index
print(f"risk_score → pos mean: {X_clean.loc[pos_risk, 'risk_score'].mean():.4f}")
print(f"risk_score → neg mean: {X_clean.loc[neg_risk, 'risk_score'].mean():.4f}")