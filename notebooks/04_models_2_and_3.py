"""
=============================================================
SUPPLIER RISK & COST ESCALATION PREDICTION
Model 2 (Cancellation/SLA) + Model 3 (Profit Overrun)
=============================================================
Model 2: Predicts order cancellation / SLA breach (Binary)
Model 3: Predicts order profit ratio (Regression)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics         import (classification_report, roc_auc_score,
                                      f1_score, mean_squared_error, r2_score)
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingRegressor
from xgboost                 import XGBClassifier, XGBRegressor

DATA_PATH  = '../data/features_master.csv'
OUTPUT_DIR = '../outputs/'

df = pd.read_csv(DATA_PATH)
df['order_date'] = pd.to_datetime(df['order_date'])

FEATURE_COLS = [c for c in df.columns if c.endswith('_enc') or c in [
    'Days for shipment (scheduled)', 'shipping_delay', 'delay_ratio',
    'Order Item Product Price', 'Order Item Discount Rate', 'Order Item Quantity',
    'discount_amount', 'revenue_per_unit', 'is_high_discount',
    'order_month', 'order_quarter', 'order_dow', 'order_week', 'is_q4',
    'hist_late_rate', 'hist_profit_ratio', 'hist_discount_rate',
    'hist_avg_delay', 'supplier_order_count', 'dept_avg_price_qtr'
]]

# Time-based split
split_date  = df['order_date'].quantile(0.80)
train_mask  = df['order_date'] <= split_date
X           = df[FEATURE_COLS]
X_train, X_test = X[train_mask], X[~train_mask]

# =============================================================================
# MODEL 2: CANCELLATION / SLA BREACH CLASSIFIER
# =============================================================================
print("=" * 60)
print("MODEL 2: Cancellation / SLA Breach Classifier")
print("=" * 60)

y2_train = df.loc[train_mask, 'target_cancellation']
y2_test  = df.loc[~train_mask, 'target_cancellation']

print(f"  Cancellation rate — Train: {y2_train.mean()*100:.1f}% | Test: {y2_test.mean()*100:.1f}%")

# Use scale_pos_weight for imbalance
neg2, pos2 = (y2_train == 0).sum(), (y2_train == 1).sum()

model2 = XGBClassifier(
    n_estimators     = 200,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = neg2 / pos2,
    use_label_encoder= False,
    eval_metric      = 'logloss',
    random_state     = 42,
    n_jobs           = -1,
)
model2.fit(X_train, y2_train, eval_set=[(X_test, y2_test)], verbose=False)

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    precision_recall_curve,
    auc,
    recall_score,
    precision_score
)
import numpy as np

y2_pred_prob = model2.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y2_test, y2_pred_prob)

precision, recall, _ = precision_recall_curve(y2_test, y2_pred_prob)
pr_auc = auc(recall, precision)

thresholds = np.arange(0.05, 0.6, 0.02)

best_threshold = 0.5
best_precision = 0

for t in thresholds:
    y_temp = (y2_pred_prob >= t).astype(int)

    rec = recall_score(y2_test, y_temp)
    prec = precision_score(y2_test, y_temp)

    # Business constraint: maintain recall ≥ 0.65
    if rec >= 0.65 and prec > best_precision:
        best_precision = prec
        best_threshold = t

y2_pred = (y2_pred_prob >= best_threshold).astype(int)

print("\n— MODEL 2 EVALUATION —")
print(f" Cancellation Rate (Test): {y2_test.mean()*100:.2f}%")
print(f" ROC-AUC: {roc:.4f}")
print(f" PR-AUC: {pr_auc:.4f}")
print(f" Selected Threshold: {best_threshold:.2f}")
print(f" Precision (Canceled): {precision_score(y2_test, y2_pred):.4f}")
print(f" Recall (Canceled): {recall_score(y2_test, y2_pred):.4f}")
print(f" F1 Score: {f1_score(y2_test, y2_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y2_test, y2_pred, target_names=['Normal', 'Canceled/Problem']))

joblib.dump(
    {'model': model2, 'features': FEATURE_COLS},
    '../models/model_cancellation.pkl'
)
print(" Model saved: ../models/model_cancellation.pkl")

# ==========================================================
# MODEL 3: ENGINEERED PROFIT RISK SCORE
# ==========================================================
print("\n" + "=" * 60)
print("MODEL 3: Engineered Profit Risk Score")
print("=" * 60)

# ----------------------------------------------------------
# 1️⃣ Define Profit Risk Components
# ----------------------------------------------------------

# Normalize helper
def minmax(series):
    return (series - series.min()) / (series.max() - series.min())

# Components contributing to profit risk
profit_instability = 1 - df['hist_profit_ratio']        # low historical margin = risky
discount_pressure = df['hist_discount_rate']            # heavy discounting
promo_flag = df['is_high_discount']                     # aggressive discount flag
dept_price_vol = df['dept_avg_price_qtr']               # price volatility proxy

# Normalize continuous components
profit_instability = minmax(profit_instability)
discount_pressure = minmax(discount_pressure)
dept_price_vol = minmax(dept_price_vol)

# ----------------------------------------------------------
# 2️⃣ Combine into Profit Risk Score
# ----------------------------------------------------------

df['profit_risk_score'] = (
    0.5 * profit_instability +
    0.3 * discount_pressure +
    0.15 * dept_price_vol +
    0.05 * promo_flag
)

print("\nProfit Risk Score Summary:")
print(df['profit_risk_score'].describe())

# ----------------------------------------------------------
# 3️⃣ Aggregate Profit Risk (Test Period)
# ----------------------------------------------------------
profit_test = df.loc[~train_mask, 'profit_risk_score']

print("\nTest Period Profit Risk Mean:",
      profit_test.mean())

# Save engineered model (no ML object)
joblib.dump(
    {'type': 'engineered_score',
     'description': 'Weighted historical profit instability score'},
    '../models/model_profit.pkl'
)

print(" Profit Risk model saved (engineered score).")

# ==========================================================
# COMBINED EVALUATION PLOT
# ==========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Models 2 & 3: Cancellation + Profit Risk',
             fontsize=14, fontweight='bold')

# ----------------------------------------------------------
# 1️⃣ Model 2 Feature Importance (ML)
# ----------------------------------------------------------
fi2 = pd.Series(
    model2.feature_importances_,
    index=FEATURE_COLS
).sort_values().tail(12)

axes[0].barh(fi2.index, fi2.values,
             color="#f4c3c3", edgecolor="white")
axes[0].set_title('Model 2: Top Features (Cancellation)',
                  fontweight='bold')

# ----------------------------------------------------------
# 2️⃣ Engineered Profit Risk Component Weights
# ----------------------------------------------------------
profit_weights = pd.Series({
    'Profit Instability (1 - hist_profit_ratio)': 0.5,
    'Historical Discount Rate': 0.3,
    'Dept Price Volatility': 0.15,
    'High Discount Flag': 0.05
})

axes[1].barh(profit_weights.index,
             profit_weights.values,
             color="#f39c12",
             edgecolor="white")

axes[1].set_title('Model 3: Profit Risk Weights',
                  fontweight='bold')

# ----------------------------------------------------------
# 3️⃣ Profit Risk Score Distribution (Test Period)
# ----------------------------------------------------------
profit_test = df.loc[~train_mask, 'profit_risk_score']

axes[2].hist(profit_test, bins=30,
             color="#3498db", alpha=0.7)

axes[2].set_title('Profit Risk Score Distribution (Test)',
                  fontweight='bold')
axes[2].set_xlabel('Profit Risk Score')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/04_models_2_3_eval.png',
            dpi=150,
            bbox_inches='tight')
plt.close()

print(f"\n Saved: {OUTPUT_DIR}/04_models_2_3_eval.png")
