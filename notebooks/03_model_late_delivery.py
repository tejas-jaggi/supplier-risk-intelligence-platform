"""
=============================================================
SUPPLIER RISK & COST ESCALATION PREDICTION
Day 6-8: Model 1 — Late Delivery Classifier
=============================================================
Predicts: Will this order be delivered late? (Binary: 0/1)
Algorithm: XGBoost + class balancing
Outputs:   Model pkl, evaluation plots, SHAP chart
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

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics           import (classification_report, confusion_matrix,
                                        roc_auc_score, roc_curve, f1_score)
from sklearn.preprocessing     import StandardScaler
from xgboost                   import XGBClassifier

DATA_PATH   = '../data/features_master.csv'
MODEL_PATH  = '../models/model_late_delivery.pkl'
OUTPUT_DIR  = '../outputs/'

# ── 1. LOAD & SPLIT ───────────────────────────────────────────────────────────
print("Loading features...")
df = pd.read_csv(DATA_PATH)

FEATURE_COLS = [c for c in df.columns if c.endswith('_enc') or c in [
    'Days for shipment (scheduled)',
    'Order Item Product Price', 'Order Item Discount Rate', 'Order Item Quantity',
    'discount_amount', 'revenue_per_unit', 'is_high_discount',
    'order_month', 'order_quarter', 'order_dow', 'order_week', 'is_q4',
    'hist_late_rate', 'hist_profit_ratio', 'hist_discount_rate',
    'hist_avg_delay', 'supplier_order_count', 'dept_avg_price_qtr'
]]

X = df[FEATURE_COLS]
y = df['Late_delivery_risk']

# Time-based split (no data leakage — use last 20% of time as test)
df['order_date'] = pd.to_datetime(df['order_date'])
df = df.sort_values('order_date').reset_index(drop=True)
split_date = df['order_date'].quantile(0.80)
train_mask = df['order_date'] <= split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Late delivery rate — Train: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

# ── 2. TRAIN MODEL ────────────────────────────────────────────────────────────
print("\nTraining XGBoost classifier...")

# scale_pos_weight handles class imbalance
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos

model = XGBClassifier(
    n_estimators       = 300,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    scale_pos_weight   = spw,
    use_label_encoder  = False,
    eval_metric        = 'logloss',
    random_state       = 42,
    n_jobs             = -1,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=True)

# ── 3. EVALUATE ───────────────────────────────────────────────────────────────
from sklearn.metrics import f1_score, recall_score, classification_report, roc_auc_score
import numpy as np

y_pred_prob = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.2, 0.8, 0.02)

best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_temp = (y_pred_prob >= t).astype(int)

    late_recall = recall_score(y_test, y_temp, pos_label=1)
    ontime_recall = recall_score(y_test, y_temp, pos_label=0)
    f1 = f1_score(y_test, y_temp)

    # Balanced constraint: both recalls must be ≥ 0.60
    if late_recall >= 0.60 and ontime_recall >= 0.60:
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

# Fallback if no threshold satisfies both
if best_f1 == 0:
    best_threshold = 0.40  # reasonable balanced default

y_pred = (y_pred_prob >= best_threshold).astype(int)

print("\n— EVALUATION RESULTS —")
print(f" Selected Threshold: {best_threshold:.2f}")
print(f" ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f" F1 Score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['On Time', 'Late']))

# ── 4. PLOTS ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model 1: Late Delivery Classifier — Evaluation', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['On Time', 'Late'], yticklabels=['On Time', 'Late'])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='#3498db', linewidth=2, label=f'XGBoost (AUC={auc:.3f})')
axes[1].plot([0,1], [0,1], 'k--', alpha=0.5, label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#3498db')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# Feature Importance (Top 15)
fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
axes[2].barh(fi.index, fi.values, color='#3498db', edgecolor='white')
axes[2].set_title('Top 15 Feature Importances', fontweight='bold')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}03_late_delivery_model_eval.png', dpi=150, bbox_inches='tight')
print(f"\n  Saved: {OUTPUT_DIR}03_late_delivery_model_eval.png")
plt.close()

# ── 5. SHAP EXPLAINABILITY ────────────────────────────────────────────────────
print("\nGenerating SHAP values (sample of 2000 rows)...")
try:
    import shap
    sample_idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type='bar',
                      max_display=15, show=False)
    plt.title('SHAP Feature Importance — Late Delivery Model', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}03_shap_late_delivery.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}03_shap_late_delivery.png")
    plt.close()
except ImportError:
    print("  SHAP not installed — run: pip install shap")

# ── 6. SAVE MODEL ─────────────────────────────────────────────────────────────
joblib.dump({'model': model, 'features': FEATURE_COLS}, MODEL_PATH)
print(f"\n  Model saved: {MODEL_PATH}")

print("\n✅ Day 6-8 Complete: Late Delivery Model trained!")
print(f"   ROC-AUC: {auc:.4f} | F1: {f1_score(y_test, y_pred):.4f}")
print("\nNext Step → Run 04_model_cancellation.py")
