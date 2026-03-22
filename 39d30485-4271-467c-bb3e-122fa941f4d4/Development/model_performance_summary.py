from sklearn.metrics import average_precision_score, precision_score, recall_score
import matplotlib.pyplot as plt

# ============================================
# TRAINING DATA SUMMARY
# ============================================
print("=" * 50)
print("TRAINING DATA SUMMARY")
print("=" * 50)
print(f"Total rows          : {len(df):,}")
print(f"Total features      : {X_clean.shape[1]}")
print(f"Positive (claim=1)  : {y.sum():,} ({y.mean()*100:.2f}%)")
print(f"Negative (claim=0)  : {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")

# ============================================
# TEST DATA SUMMARY
# ============================================
print("\n" + "=" * 50)
print("TEST DATA SUMMARY")
print("=" * 50)
print(f"Total rows          : {len(test):,}")
print(f"Total features      : {X_test.shape[1]}")
print(f"Avg predicted prob  : {final_test_preds.mean():.4f}")
print(f"Min predicted prob  : {final_test_preds.min():.4f}")
print(f"Max predicted prob  : {final_test_preds.max():.4f}")
print(f"Predicted positive  : {(final_test_preds > 0.5).sum():,} ({(final_test_preds > 0.5).mean()*100:.2f}%)")

# ============================================
# MODEL PERFORMANCE SUMMARY
# ============================================
print("\n" + "=" * 50)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 50)
print(f"LightGBM  CV PR-AUC : 0.06453")
print(f"CatBoost  CV PR-AUC : 0.06583")
print(f"Ensemble  CV PR-AUC : 0.06621")
print(f"Best weights        : LGB 0.3 + CAT 0.7")

# ============================================
# PLOT PREDICTION DISTRIBUTION
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training class distribution
axes[0].bar(['Negative (0)', 'Positive (1)'],
            [(y==0).sum(), y.sum()],
            color=['steelblue', '#e85d04'])
axes[0].set_title('Training Data - Class Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate([(y==0).sum(), y.sum()]):
    axes[0].text(i, v + 1000, f'{v:,}\n({v/len(y)*100:.1f}%)',
                ha='center', fontweight='bold')

# Plot 2: Test prediction distribution
axes[1].hist(final_test_preds, bins=50, color='steelblue', edgecolor='white')
axes[1].set_title('Test Data - Predicted Probability Distribution')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Count')
axes[1].axvline(x=0.5, color='red', linestyle='--', label='threshold=0.5')
axes[1].legend()

plt.tight_layout()
plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDashboard saved as summary_dashboard.png")
print("\nSubmission file: submission.csv")
print("Rows in submission:", len(submission))