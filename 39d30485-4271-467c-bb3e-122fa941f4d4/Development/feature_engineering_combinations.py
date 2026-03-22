# Try adding more feature combinations
X_clean['f22_x_f11_x_f21'] = X_clean['feature_22'] * X_clean['feature_11'] * X_clean['feature_21']
X_clean['f16_x_f11_x_f21'] = X_clean['feature_16'] * X_clean['feature_11'] * X_clean['feature_21']
X_clean['risk_x_f24'] = X_clean['risk_score'] * X_clean['feature_24']
X_clean['risk_x_f10'] = X_clean['risk_score'] * X_clean['feature_10']
X_clean['f32_x_f22'] = X_clean['feature_32'] * X_clean['feature_22']
X_clean['f32_x_f11'] = X_clean['feature_32'] * X_clean['feature_11']

# Check correlations
new_feats = ['f22_x_f11_x_f21', 'f16_x_f11_x_f21', 
             'risk_x_f24', 'risk_x_f10',
             'f32_x_f22', 'f32_x_f11']

for col in new_feats:
    corr = abs(X_clean[col].corr(y))
    print(f"{col}: {corr:.5f}")

print("\nX_clean shape:", X_clean.shape)