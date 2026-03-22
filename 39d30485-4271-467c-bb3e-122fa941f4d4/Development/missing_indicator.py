
# 1. Add missing flags
for col in missing_cols:
    X[col + "_missing"] = X[col].isnull().astype(int)

# 2. Special handling for feature_8 (high missing)
if 'feature_8' in missing_cols:
    X['feature_8'] = X['feature_8'].fillna(-999)

# 3. Fill remaining columns with median
for col in missing_cols:
    if col != 'feature_8':
        X[col] = X[col].fillna(X[col].median())

# 4. Final check
print("Total missing:", X.isnull().sum().sum())

print(X.shape)