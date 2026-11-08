import os
import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)

print("Starting data cleaning pipeline...")

# ==================== 1. CLEAN CUSTOMERS ====================
cust = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
cust["city"] = cust["city"].str.strip().str.title()

def clean_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g_str = str(g).strip().lower()
    if g_str in ["m", "male"]:
        return "male"
    elif g_str in ["f", "female"]:
        return "female"
    return "unknown"

cust["gender"] = cust["gender"].apply(clean_gender)

cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
city_median_ages = cust.groupby("city")["age"].transform("median")
cust["age"] = cust["age"].fillna(city_median_ages)

cust["signup_date"] = pd.to_datetime(cust["signup_date"])
cust.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
print("-> customers.csv cleaned and saved.")

# ==================== 2. PRODUCTS ====================
prod = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
prod.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
print("-> products.csv saved.")

# ==================== 3. CLEAN ORDERS ====================
orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders = orders.drop_duplicates()
orders = orders.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")

orders = orders.merge(prod[["product_id", "price_inr", "category"]], on="product_id", how="left")

orders["order_amount_num"] = pd.to_numeric(orders["order_amount"], errors="coerce")
calculated_amount = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount_num"].fillna(calculated_amount)
orders = orders.drop(columns=["order_amount_num"])

orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

completed_mask = orders["status"] == "completed"
category_medians = orders[completed_mask].groupby("category")["delivery_minutes"].transform("median")
orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(category_medians)

orders = orders[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]]
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

orders.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)
print("-> orders.csv cleaned and saved.")
print("Data cleaning completed successfully!")
