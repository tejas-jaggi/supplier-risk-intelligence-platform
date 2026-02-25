"""
=============================================================
SUPPLIER RISK & COST ESCALATION PREDICTION
Day 4-5: Feature Engineering
=============================================================
Builds the master feature table used by all 3 models.
Output: ../data/features_master.csv
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_PATH   = '../data/DataCoSupplyChainDataset.csv'
OUTPUT_PATH = '../data/features_master.csv'

print("Loading data...")
df = pd.read_csv(DATA_PATH, encoding='latin1')
df['order_date']    = pd.to_datetime(df['order date (DateOrders)'])
df['shipping_date'] = pd.to_datetime(df['shipping date (DateOrders)'])

# ── 1. TIME FEATURES ──────────────────────────────────────────────────────────
df['order_year']     = df['order_date'].dt.year
df['order_month']    = df['order_date'].dt.month
df['order_quarter']  = df['order_date'].dt.quarter
df['order_dow']      = df['order_date'].dt.dayofweek
df['order_week']     = df['order_date'].dt.isocalendar().week.astype(int)
df['quarter_label']  = df['order_date'].dt.to_period('Q').astype(str)
df['is_q4']          = (df['order_quarter'] == 4).astype(int)   # Holiday season flag

# ── 2. DELIVERY & DELAY FEATURES ─────────────────────────────────────────────
df['shipping_delay']     = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
df['delay_ratio']        = df['shipping_delay'] / (df['Days for shipment (scheduled)'] + 0.01)
df['sched_days_bin']     = pd.cut(df['Days for shipment (scheduled)'], bins=[0,1,3,6,999],
                                   labels=['same_day','fast','standard','slow'])

# ── 3. FINANCIAL FEATURES ─────────────────────────────────────────────────────
df['discount_amount']    = df['Order Item Product Price'] * df['Order Item Discount Rate']
df['revenue_per_unit']   = df['Sales'] / (df['Order Item Quantity'] + 0.01)
df['profit_per_unit']    = df['Order Profit Per Order'] / (df['Order Item Quantity'] + 0.01)
df['is_high_discount']   = (df['Order Item Discount Rate'] > 0.20).astype(int)
df['is_negative_profit'] = (df['Order Item Profit Ratio'] < 0).astype(int)
df['is_low_profit']      = (df['Order Item Profit Ratio'] < df['Order Item Profit Ratio'].quantile(0.25)).astype(int)

# ── 4. SUPPLIER-LEVEL ROLLING FEATURES ───────────────────────────────────────
# Aggregate per Department × Category (supplier proxy)
# Sort by date to compute rolling/historical averages without data leakage
df = df.sort_values('order_date').reset_index(drop=True)

supplier_key = ['Department Name', 'Category Name']

# Expanding mean (historical average up to but not including current row)
# This simulates knowing a supplier's historical performance at prediction time
for col, alias in [
    ('Late_delivery_risk',     'hist_late_rate'),
    ('Order Item Profit Ratio','hist_profit_ratio'),
    ('Order Item Discount Rate','hist_discount_rate'),
    ('shipping_delay',         'hist_avg_delay'),
]:
    df[alias] = (
        df.groupby(supplier_key)[col]
          .transform(lambda x: x.expanding().mean().shift(1))
          .fillna(df[col].mean())  # fill NaN for first occurrence
    )

# Supplier order volume (as of current order)
df['supplier_order_count'] = (
    df.groupby(supplier_key).cumcount() + 1
)

# ── 5. QUARTERLY PRICE CHANGE FEATURE ────────────────────────────────────────
qtr_price = df.groupby(['Department Name', 'quarter_label'])['Order Item Product Price'].mean()
qtr_price_df = qtr_price.reset_index()
qtr_price_df.columns = ['Department Name', 'quarter_label', 'dept_avg_price_qtr']
df = df.merge(qtr_price_df, on=['Department Name', 'quarter_label'], how='left')

# ── 6. CATEGORICAL ENCODING ───────────────────────────────────────────────────
# Label encode categoricals
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Type', 'Market', 'Order Region', 'Shipping Mode',
            'Department Name', 'Category Name', 'Customer Segment',
            'sched_days_bin']

le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    le_map[col] = le

# ── 7. TARGET VARIABLES ───────────────────────────────────────────────────────
# Already exists: Late_delivery_risk (0/1)
df['target_cancellation'] = df['Order Status'].isin(
    ['CANCELED', 'SUSPECTED_FRAUD', 'PAYMENT_REVIEW']
).astype(int)
# target_profit = Order Item Profit Ratio (continuous, for regression)

# ── 8. FINAL FEATURE SET ─────────────────────────────────────────────────────
FEATURE_COLS = [
    # Shipping / delivery
    'Days for shipment (scheduled)', 'shipping_delay', 'delay_ratio',
    # Financial
    'Order Item Product Price', 'Order Item Discount Rate', 'Order Item Quantity',
    'discount_amount', 'revenue_per_unit', 'is_high_discount',
    # Time
    'order_month', 'order_quarter', 'order_dow', 'order_week', 'is_q4',
    # Historical supplier performance
    'hist_late_rate', 'hist_profit_ratio', 'hist_discount_rate',
    'hist_avg_delay', 'supplier_order_count',
    # Price signal
    'dept_avg_price_qtr',
    # Encoded categoricals
    'Type_enc', 'Market_enc', 'Order Region_enc', 'Shipping Mode_enc',
    'Department Name_enc', 'Category Name_enc', 'Customer Segment_enc',
]

TARGET_COLS = [
    'Late_delivery_risk',       # Model 1
    'target_cancellation',      # Model 2
    'Order Item Profit Ratio',  # Model 3
]

master = df[FEATURE_COLS + TARGET_COLS + ['Order Id', 'order_date', 'quarter_label',
                                           'Department Name', 'Category Name']].copy()
master.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Feature engineering complete!")
rows, cols = master.shape
print(f" Rows: {rows:,}")
print(f" Columns: {cols}")
print(f"   Feature columns: {len(FEATURE_COLS)}")
print(f"   Target columns: {len(TARGET_COLS)}")
print(f"   Saved to: {OUTPUT_PATH}")
print(f"\n   Target summary:")
print(f"   • Late delivery rate:   {master['Late_delivery_risk'].mean()*100:.1f}%")
print(f"   • Cancellation rate:    {master['target_cancellation'].mean()*100:.1f}%")
print(f"   • Avg profit ratio:     {master['Order Item Profit Ratio'].mean()*100:.1f}%")
print(f" Missing values in master: {master.isnull().sum().sum()}")
print("\nNext Step → Run 03_model_late_delivery.py")
