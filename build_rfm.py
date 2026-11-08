import os
import pandas as pd

# Define paths and snapshot date
CLEAN_DIR = "data/clean"
OUTPUT_DIR = "data/clean"
SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

print("Loading clean orders and customers datasets...")
orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))

# Convert order_ts to datetime format
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

print("Filtering for completed orders only...")
completed_orders = orders[orders["status"] == "completed"].copy()

print("Computing RFM metrics per customer...")
# Group by customer_id
rfm = completed_orders.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency in whole days from the snapshot date
rfm["recency"] = (SNAPSHOT_DATE - rfm["last_order_ts"]).dt.days

# Drop the temporary timestamp column
rfm = rfm.drop(columns=["last_order_ts"])

# Merge with customer demographic details if needed
rfm_full = customers.merge(rfm, on="customer_id", how="inner")

# Save the resulting RFM table
output_path = os.path.join(OUTPUT_DIR, "rfm_table.csv")
rfm_full.to_csv(output_path, index=False)
print(f"-> RFM table successfully saved to {output_path} ({len(rfm_full)} customers)")

# ==========================================
# Verification against Reference Checkpoints
# ==========================================
print("\n--- Verification Checkpoints ---")
print(f"Total Customers with completed orders: {len(rfm_full)} (Expected: 1,175)")
print(f"Recency Median: {rfm_full['recency'].median():.0f} (Expected: 19)")
print(f"Recency Min: {rfm_full['recency'].min()} | Max: {rfm_full['recency'].max()} (Expected: 0 / 180)")
print(f"Frequency Mean: {rfm_full['frequency'].mean():.2f} | Max: {rfm_full['frequency'].max()} (Expected: 9.46 / 33)")
print(f"Monetary Mean: INR {rfm_full['monetary'].mean():,.2f} | Max: INR {rfm_full['monetary'].max():,.2f} (Expected: INR 2,552.06 / INR 14,766.55)")
