import pandas as pd

# Load dataset (use features_master since that's what models use)
df = pd.read_csv("../data/features_master.csv")

# Recreate low_profit_risk target
df['low_profit_risk'] = (df['Order Item Profit Ratio'] < 0.10).astype(int)

# Recreate FEATURE_COLS (copy exact logic)
FEATURE_COLS = [
    c for c in df.columns
    if c.endswith('_enc') or c in [
        'Days for shipment (scheduled)',
        'Order Item Product Price',
        'Order Item Discount Rate',
        'Order Item Quantity',
        'discount_amount',
        'revenue_per_unit',
        'is_high_discount',
        'order_month', 'order_quarter', 'order_dow', 'order_week', 'is_q4',
        'hist_late_rate', 'hist_profit_ratio', 'hist_discount_rate',
        'hist_avg_delay', 'supplier_order_count', 'dept_avg_price_qtr'
    ]
]

# Compute correlations
corrs = (
    df[FEATURE_COLS + ['low_profit_risk']]
    .corr()['low_profit_risk']
    .sort_values(ascending=False)
)

print("\nTop correlations with low_profit_risk:")
print(corrs.head(10))

print("\nLowest correlations:")
print(corrs.tail(10))