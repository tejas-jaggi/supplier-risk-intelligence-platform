# ==========================================================
# 05: FINAL SUPPLIER RISK SCORING
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

print("Loading features & models...")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features_master.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------------------------------------
# LOAD MODELS (ONLY ML MODELS)
# ----------------------------------------------------------
MODEL1_PATH = os.path.join(PROJECT_ROOT, "models", "model_late_delivery.pkl")
MODEL2_PATH = os.path.join(PROJECT_ROOT, "models", "model_cancellation.pkl")

m1 = joblib.load(MODEL1_PATH)
m2 = joblib.load(MODEL2_PATH)

model1, feat1 = m1['model'], m1['features']
model2, feat2 = m2['model'], m2['features']

# ----------------------------------------------------------
# GENERATE ML PROBABILITIES
# ----------------------------------------------------------
df['pred_late_prob'] = model1.predict_proba(df[feat1])[:, 1]
df['pred_cancel_prob'] = model2.predict_proba(df[feat2])[:, 1]

# ----------------------------------------------------------
# ENGINEERED PROFIT RISK (NO ML)
# ----------------------------------------------------------
def minmax(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9)

profit_instability = 1 - df['hist_profit_ratio']
discount_pressure = df['hist_discount_rate']
dept_price_vol = df['dept_avg_price_qtr']
promo_flag = df['is_high_discount']

profit_instability = minmax(profit_instability)
discount_pressure = minmax(discount_pressure)
dept_price_vol = minmax(dept_price_vol)

df['pred_profit_risk'] = (
    0.5 * profit_instability +
    0.3 * discount_pressure +
    0.15 * dept_price_vol +
    0.05 * promo_flag
)

# ----------------------------------------------------------
# COMPOSITE RISK SCORE (40 / 35 / 25)
# ----------------------------------------------------------
df['composite_risk'] = (
    0.40 * df['pred_late_prob'] +
    0.35 * df['pred_cancel_prob'] +
    0.25 * df['pred_profit_risk']
)

print("\nComposite Risk Summary:")
print(df['composite_risk'].describe())

# ----------------------------------------------------------
# SUPPLIER-LEVEL AGGREGATION
# ----------------------------------------------------------
supplier_summary = (
    df.groupby(['Department Name', 'Category Name'])
      .agg(
          total_orders=('Order Id', 'count'),
          avg_late_prob=('pred_late_prob', 'mean'),
          avg_cancel_prob=('pred_cancel_prob', 'mean'),
          avg_profit_risk=('pred_profit_risk', 'mean'),
          composite_risk=('composite_risk', 'mean')
      )
      .reset_index()
)

supplier_summary = supplier_summary.sort_values(
    'composite_risk', ascending=False
)

supplier_summary['risk_rank'] = (
    supplier_summary['composite_risk']
    .rank(ascending=False, method='dense')
    .astype(int)
)

# ----------------------------------------------------------
# RISK TIERS
# ----------------------------------------------------------
supplier_summary['risk_tier'] = pd.qcut(
    supplier_summary['composite_risk'],
    q=4,
    labels=['Low', 'Moderate', 'High', 'Critical']
)

print("\nTop 5 Highest Risk Suppliers:")
print(supplier_summary.head())

print("\nRisk Tier Distribution:")
print(supplier_summary['risk_tier'].value_counts())

# ----------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------
supplier_summary.to_csv(
    f"{OUTPUT_DIR}/final_supplier_risk_summary.csv",
    index=False
)

print("\nSaved: final_supplier_risk_summary.csv")

# ----------------------------------------------------------
# SIMPLE DASHBOARD PLOT
# ----------------------------------------------------------
plt.figure(figsize=(8, 5))
supplier_summary['risk_tier'].value_counts().plot(
    kind='bar',
    color=['green', 'orange', 'red', 'darkred']
)
plt.title("Supplier Risk Tier Distribution")
plt.ylabel("Number of Suppliers")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_risk_tier_distribution.png", dpi=150)
plt.close()

print("Saved: 05_risk_tier_distribution.png")

print("\nFINAL SUPPLIER RISK SCORING COMPLETE")