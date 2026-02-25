"""
=============================================================
SUPPLIER RISK & COST ESCALATION PREDICTION
Day 1-2: Exploratory Data Analysis
=============================================================
Run this file with: python 01_EDA.py
Or paste sections into Jupyter notebook cell by cell.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Remove this line if running in Jupyter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH  = '../data/DataCoSupplyChainDataset.csv'
OUTPUT_DIR = '../outputs/'

sns.set_style("whitegrid")
sns.set_palette("husl")

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding='latin1')
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. BASIC PROFILE ──────────────────────────────────────────────────────────
print("\n── BASIC PROFILE ──")
print(f"  Date range: {df['order date (DateOrders)'].min()} → {df['order date (DateOrders)'].max()}")
print(f"  Delivery Status:\n{df['Delivery Status'].value_counts()}")
print(f"\n  Missing values (only cols with missing):")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ── 3. FEATURE ENGINEERING (foundational) ─────────────────────────────────────
print("\n── FEATURE ENGINEERING ──")

df['order_date']     = pd.to_datetime(df['order date (DateOrders)'])
df['shipping_date']  = pd.to_datetime(df['shipping date (DateOrders)'])

# Core delay feature
df['shipping_delay'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

# Time features
df['order_year']    = df['order_date'].dt.year
df['order_month']   = df['order_date'].dt.month
df['order_quarter'] = df['order_date'].dt.quarter
df['order_dow']     = df['order_date'].dt.dayofweek   # 0=Monday
df['quarter_label'] = df['order_date'].dt.to_period('Q').astype(str)

# Profit flag (bottom 25% = at-risk)
profit_q25 = df['Order Item Profit Ratio'].quantile(0.25)
df['low_profit_flag'] = (df['Order Item Profit Ratio'] < profit_q25).astype(int)

# Order cancelled / problematic flag
df['order_problem_flag'] = df['Order Status'].isin(
    ['CANCELED', 'SUSPECTED_FRAUD', 'PAYMENT_REVIEW']
).astype(int)

# Discount pressure flag
df['high_discount_flag'] = (df['Order Item Discount Rate'] > 0.2).astype(int)

print(f"  Late delivery rate:  {df['Late_delivery_risk'].mean()*100:.1f}%")
print(f"  Low profit rate:     {df['low_profit_flag'].mean()*100:.1f}%")
print(f"  Problem order rate:  {df['order_problem_flag'].mean()*100:.1f}%")
print(f"  Avg shipping delay:  {df['shipping_delay'].mean():.2f} days")

# ── 4. SUPPLIER-LEVEL AGGREGATION ─────────────────────────────────────────────
# Treat Department + Category as "supplier" proxy
print("\n── SUPPLIER-LEVEL RISK SUMMARY ──")

supplier_summary = df.groupby(['Department Name', 'Category Name']).agg(
    total_orders        = ('Order Id',               'count'),
    late_delivery_rate  = ('Late_delivery_risk',     'mean'),
    avg_profit_ratio    = ('Order Item Profit Ratio','mean'),
    avg_discount_rate   = ('Order Item Discount Rate','mean'),
    avg_shipping_delay  = ('shipping_delay',          'mean'),
    problem_order_rate  = ('order_problem_flag',      'mean'),
    total_revenue       = ('Sales',                   'sum'),
    total_profit        = ('Order Profit Per Order',  'sum'),
).reset_index()

# Composite Risk Score (simple weighted average, to be refined in Phase 4)
# Higher score = higher risk
supplier_summary['risk_score'] = (
    supplier_summary['late_delivery_rate']   * 0.40 +
    supplier_summary['problem_order_rate']   * 0.30 +
    (1 - supplier_summary['avg_profit_ratio'].clip(0, 1)) * 0.20 +
    supplier_summary['avg_discount_rate']    * 0.10
).round(4)

supplier_summary = supplier_summary.sort_values('risk_score', ascending=False)

print("\nTop 10 Highest Risk Supplier Categories:")
print(supplier_summary[['Department Name','Category Name','total_orders',
                         'late_delivery_rate','avg_profit_ratio','risk_score']].head(10).to_string(index=False))

# Save summary
supplier_summary.to_csv(f'{OUTPUT_DIR}supplier_risk_summary.csv', index=False)
print(f"\n  Saved to {OUTPUT_DIR}supplier_risk_summary.csv")

# ── 5. QUARTERLY PRICE ESCALATION SIGNAL ─────────────────────────────────────
print("\n── PRICE ESCALATION SIGNAL ──")

qtr_price = df.groupby(['Department Name', 'quarter_label'])['Order Item Product Price'].mean().reset_index()
qtr_price = qtr_price.pivot(index='quarter_label', columns='Department Name', values='Order Item Product Price')
qtr_price_pct_change = qtr_price.pct_change() * 100

print("Avg quarterly price change by department (%):")
print(qtr_price_pct_change.mean().sort_values(ascending=False).round(2))

# ── 6. EDA VISUALIZATIONS ─────────────────────────────────────────────────────
print("\n── GENERATING VISUALIZATIONS ──")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Supply Chain Risk & Performance — EDA Overview', fontsize=16, fontweight='bold')

# Plot 1: Delivery Status Distribution
palette = {'Late delivery':'#e74c3c','Advance shipping':'#2ecc71',
           'Shipping on time':'#3498db','Shipping canceled':'#e67e22'}
counts = df['Delivery Status'].value_counts()
colors = [palette.get(s, '#95a5a6') for s in counts.index]
axes[0,0].bar(counts.index, counts.values, color=colors, edgecolor='white')
axes[0,0].set_title('Delivery Status Distribution', fontweight='bold')
axes[0,0].set_ylabel('Order Count')
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v + 400, f'{v/len(df)*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[0,0].tick_params(axis='x', rotation=12)

# Plot 2: Late Delivery Rate by Department
late_by_dept = df.groupby('Department Name')['Late_delivery_risk'].mean().sort_values() * 100
c2 = ['#e74c3c' if v > 55 else '#e67e22' if v > 50 else '#2ecc71' for v in late_by_dept.values]
axes[0,1].barh(late_by_dept.index, late_by_dept.values, color=c2, edgecolor='white')
axes[0,1].axvline(late_by_dept.mean(), color='black', linestyle='--', alpha=0.6,
                  label=f'Avg: {late_by_dept.mean():.1f}%')
axes[0,1].set_title('Late Delivery Rate by Department\n(Supplier Proxy)', fontweight='bold')
axes[0,1].set_xlabel('Late Delivery Rate (%)')
axes[0,1].legend(fontsize=9)
for i, v in enumerate(late_by_dept.values):
    axes[0,1].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=9)

# Plot 3: Shipping Delay Distribution
axes[0,2].hist(df.loc[df['Late_delivery_risk']==0, 'shipping_delay'].clip(-3,4),
               bins=15, alpha=0.6, color='#2ecc71', label='On Time / Early', edgecolor='white')
axes[0,2].hist(df.loc[df['Late_delivery_risk']==1, 'shipping_delay'].clip(-3,4),
               bins=15, alpha=0.6, color='#e74c3c', label='Late', edgecolor='white')
axes[0,2].axvline(0, color='black', linestyle='--', alpha=0.7)
axes[0,2].set_title('Shipping Delay Distribution\n(Actual − Scheduled Days)', fontweight='bold')
axes[0,2].set_xlabel('Delay (Days)')
axes[0,2].set_ylabel('Count')
axes[0,2].legend()

# Plot 4: Profit Ratio by Department
profit_by_dept = df.groupby('Department Name')['Order Item Profit Ratio'].mean().sort_values()
c4 = ['#e74c3c' if v < 0 else '#e67e22' if v < 0.1 else '#2ecc71' for v in profit_by_dept.values]
axes[1,0].barh(profit_by_dept.index, profit_by_dept.values * 100, color=c4, edgecolor='white')
axes[1,0].axvline(0, color='black', linestyle='--', alpha=0.7)
axes[1,0].set_title('Avg Profit Ratio by Department\n(Cost Overrun Risk Proxy)', fontweight='bold')
axes[1,0].set_xlabel('Avg Profit Ratio (%)')

# Plot 5: Quarterly Trends
qtr_late   = df.groupby('quarter_label')['Late_delivery_risk'].mean() * 100
qtr_profit = df.groupby('quarter_label')['Order Item Profit Ratio'].mean() * 100
ax5b = axes[1,1].twinx()
axes[1,1].plot(range(len(qtr_late)), qtr_late.values, 'o-', color='#e74c3c',
               linewidth=2, markersize=4, label='Late Delivery %')
ax5b.plot(range(len(qtr_profit)), qtr_profit.values, 's--', color='#3498db',
          linewidth=2, markersize=4, label='Profit Ratio %')
axes[1,1].set_xticks(range(len(qtr_late)))
axes[1,1].set_xticklabels(qtr_late.index, rotation=45, fontsize=7)
axes[1,1].set_ylabel('Late Delivery Rate (%)', color='#e74c3c')
ax5b.set_ylabel('Avg Profit Ratio (%)', color='#3498db')
axes[1,1].set_title('Quarterly Trends\nLate Delivery % vs Profit Ratio %', fontweight='bold')
l1, lb1 = axes[1,1].get_legend_handles_labels()
l2, lb2 = ax5b.get_legend_handles_labels()
axes[1,1].legend(l1+l2, lb1+lb2, fontsize=8, loc='upper left')

# Plot 6: Supplier Risk Matrix
rm = df.groupby('Department Name').agg(
    late_rate    = ('Late_delivery_risk',      'mean'),
    avg_profit   = ('Order Item Profit Ratio', 'mean'),
    order_volume = ('Order Id',                'count')
).reset_index()
sc = axes[1,2].scatter(rm['late_rate']*100, rm['avg_profit']*100,
                        s=rm['order_volume']/50, c=rm['late_rate'],
                        cmap='RdYlGn_r', alpha=0.85, edgecolors='white', linewidth=1)
axes[1,2].set_title('Supplier Risk Matrix\n(Bubble size = Order Volume)', fontweight='bold')
axes[1,2].set_xlabel('Late Delivery Rate (%)')
axes[1,2].set_ylabel('Avg Profit Ratio (%)')
axes[1,2].axvline(50, color='gray', linestyle='--', alpha=0.5)
axes[1,2].axhline(rm['avg_profit'].mean()*100, color='gray', linestyle='--', alpha=0.5)
for _, row in rm.iterrows():
    axes[1,2].annotate(row['Department Name'],
                       (row['late_rate']*100, row['avg_profit']*100),
                       xytext=(5, 3), textcoords='offset points', fontsize=7)
plt.colorbar(sc, ax=axes[1,2], label='Late Rate')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}01_eda_overview.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}01_eda_overview.png")
plt.close()

# ── 7. SHIPPING MODE ANALYSIS ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Shipping Mode Analysis', fontsize=14, fontweight='bold')

mode_late = df.groupby('Shipping Mode')['Late_delivery_risk'].mean().sort_values(ascending=False) * 100
axes2[0].bar(mode_late.index, mode_late.values, color=['#e74c3c','#e67e22','#f1c40f','#2ecc71'], edgecolor='white')
axes2[0].set_title('Late Delivery Rate by Shipping Mode', fontweight='bold')
axes2[0].set_ylabel('Late Delivery Rate (%)')
for i, v in enumerate(mode_late.values):
    axes2[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

mode_profit = df.groupby('Shipping Mode')['Order Item Profit Ratio'].mean().sort_values() * 100
colors_p = ['#e74c3c' if v < 10 else '#e67e22' if v < 15 else '#2ecc71' for v in mode_profit.values]
axes2[1].bar(mode_profit.index, mode_profit.values, color=colors_p, edgecolor='white')
axes2[1].set_title('Avg Profit Ratio by Shipping Mode', fontweight='bold')
axes2[1].set_ylabel('Avg Profit Ratio (%)')

plt.tight_layout()
fig2.savefig(f'{OUTPUT_DIR}01_shipping_mode_analysis.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}01_shipping_mode_analysis.png")
plt.close()

print("\n✅ Day 1–2 EDA Complete!")
print("   Key findings:")

late_rate = df['Late_delivery_risk'].mean() * 100
avg_profit = df['Order Item Profit Ratio'].mean() * 100

mode_late = df.groupby('Shipping Mode')['Late_delivery_risk'].mean().sort_values(ascending=False)
top_mode = mode_late.index[0]
top_rate = mode_late.iloc[0] * 100

print(f"   • {late_rate:.1f}% of orders are late — strong imbalance signal")
print(f"   • {top_mode} shipping has the highest late rate ({top_rate:.1f}%)")
print(f"   • Fan Shop & Apparel dominate volume")
print(f"   • Avg profit ratio: {avg_profit:.1f}% — tight margins = escalation risk")

print("\nNext Step → Run 02_feature_engineering.py")
