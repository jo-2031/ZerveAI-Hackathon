import scipy.special as sp
from sklearn.metrics import average_precision_score

lgb_logit = sp.logit(oof_preds.clip(1e-6, 1-1e-6))
cat_logit = sp.logit(cat_oof.clip(1e-6, 1-1e-6))

best_score = 0
best_w = 0.5

for lgb_w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    cat_w = 1 - lgb_w
    blend = sp.expit(lgb_w * lgb_logit + cat_w * cat_logit)
    score = average_precision_score(y, blend)
    print(f"LGB {lgb_w} + CAT {cat_w}: {score:.5f}")
    if score > best_score:
        best_score = score
        best_w = lgb_w

print(f"\nBest score: {best_score:.5f} at LGB weight {best_w}")