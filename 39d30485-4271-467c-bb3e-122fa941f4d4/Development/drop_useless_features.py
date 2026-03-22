# Drop useless new features
X_clean.drop(columns=['f22_x_f11_x_f21', 'f16_x_f11_x_f21', 
                       'f32_x_f22', 'f32_x_f11', 'risk_x_f10'], inplace=True)

print("Final shape:", X_clean.shape)

# Verify risk_x_f24 is still there
print("risk_x_f24 corr:", abs(X_clean['risk_x_f24'].corr(y)))