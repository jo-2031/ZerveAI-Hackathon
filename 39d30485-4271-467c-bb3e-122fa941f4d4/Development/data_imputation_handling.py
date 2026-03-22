X_clean = df.drop(columns=['id', 'target']).copy()

missing_cols = ['feature_8', 'feature_9', 'feature_12', 'feature_15',
                'feature_28', 'feature_29', 'feature_31', 'feature_34',
                'feature_35', 'feature_38', 'feature_39', 'feature_42', 'feature_45']

for col in missing_cols:
    X_clean[col + '_missing'] = X_clean[col].isnull().astype(int)

X_clean['feature_8'] = X_clean['feature_8'].fillna(-999)
for col in missing_cols:
    if col != 'feature_8':
        X_clean[col] = X_clean[col].fillna(X_clean[col].median())

print("Shape:", X_clean.shape)
print("Missing:", X_clean.isnull().sum().sum())