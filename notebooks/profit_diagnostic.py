import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/features_master.csv")

# Describe target
print("\n=== PROFIT RATIO SUMMARY ===")
print(df['Order Item Profit Ratio'].describe())

# Check extreme values
print("\nMin value:", df['Order Item Profit Ratio'].min())
print("Max value:", df['Order Item Profit Ratio'].max())

# Plot histogram
plt.hist(df['Order Item Profit Ratio'], bins=50)
plt.title("Profit Ratio Distribution")
plt.xlabel("Order Item Profit Ratio")
plt.ylabel("Frequency")
plt.show()