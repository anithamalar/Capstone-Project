import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)

print("Loading raw orders and products data...")
orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
prod = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))

# Rule 1: Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Rule 2: Remove exact duplicate rows
orders = orders.drop_duplicates()

# Rule 3: Keep only the record with the latest order_ts per order_id (resolving partial duplicates/status changes)
orders = orders.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")

# Merge with products to get price_inr and category for calculations
orders = orders.merge(prod[["product_id", "price_inr", "category"]], on="product_id", how="left")

# Rule 4 & 5: Convert order_amount to numeric, turning unparseable entries into missing, then fill using formula
orders["order_amount_num"] = pd.to_numeric(orders["order_amount"], errors="coerce")
calculated_amount = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount_num"].fillna(calculated_amount)
orders = orders.drop(columns=["order_amount_num"])

# Rule 6: Set negative delivery_minutes to missing, then fill completed orders using category median
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

completed_mask = orders["status"] == "completed"
category_medians = orders[completed_mask].groupby("category")["delivery_minutes"].transform("median")
orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(category_medians)

# Select final columns and format timestamp back to string
orders = orders[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]]
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Save cleaned orders
output_path = os.path.join(CLEAN_DIR, "orders.csv")
orders.to_csv(output_path, index=False)
print(f"-> Cleaned orders saved to {output_path}")

print("\n==================== CHECKPOINT VERIFICATION ====================")
print(f"Total rows: {len(orders)} (Expected: 12309)")
print("Status counts:")
print(orders["status"].value_counts())
print(f"Pending rows count: {(orders['status'] == 'pending').sum()} (Expected: 0)")
print(f"Missing order amounts: {orders['order_amount'].isnull().sum()} (Expected: 0)")

completed_sum = orders[orders["status"] == "completed"]["order_amount"].sum()
print(f"Completed orders amount sum: INR {completed_sum:,.2f} (Expected: INR 2,998,668.80)")
print("=================================================================")
