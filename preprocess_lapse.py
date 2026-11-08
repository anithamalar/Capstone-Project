import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# 1. Load clean datasets
CLEAN_DIR = "data/clean"
print("Loading clean datasets...")
orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# 2. Rebuild Cohort and Features (same logic as Task 4)
CUTOFF_DATE = pd.Timestamp("2025-09-01 00:00:00")
LAPSE_WINDOW_END = pd.Timestamp("2025-10-01 00:00:00")

orders_before_cutoff = orders[orders["order_ts"] < CUTOFF_DATE].copy()
completed_before = orders_before_cutoff[orders_before_cutoff["status"] == "completed"]
cohort_customer_ids = completed_before["customer_id"].unique()

sept_orders = orders[
    (orders["order_ts"] >= CUTOFF_DATE) & 
    (orders["order_ts"] < LAPSE_WINDOW_END) & 
    (orders["status"] == "completed")
]
active_in_sept = sept_orders["customer_id"].unique()

labels_df = pd.DataFrame({"customer_id": cohort_customer_ids})
labels_df["lapsed"] = labels_df["customer_id"].apply(lambda cid: 0 if cid in active_in_sept else 1)

orders_prod = orders_before_cutoff.merge(products, on="product_id", how="left")

recency_df = completed_before.groupby("customer_id")["order_ts"].max().reset_index()
recency_df["recency"] = (CUTOFF_DATE - recency_df["order_ts"]).dt.days
recency_df = recency_df[["customer_id", "recency"]]

completed_metrics = completed_before.groupby("customer_id").agg(
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    average_discount_pct=("discount_pct", "mean"),
    average_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

all_final_orders = orders_before_cutoff[orders_before_cutoff["status"].isin(["completed", "cancelled"])]
cancel_counts = all_final_orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    cancelled_orders=("status", lambda s: (s == "cancelled").sum())
).reset_index()
cancel_counts["cancellation_rate"] = cancel_counts["cancelled_orders"] / cancel_counts["total_orders"]
cancel_counts = cancel_counts[["customer_id", "cancellation_rate"]]

cat_counts = orders_prod.groupby(["customer_id", "category"]).size().reset_index(name="count")
most_freq_cat = cat_counts.sort_values(["customer_id", "count"], ascending=[True, False]).drop_duplicates("customer_id")
most_freq_cat = most_freq_cat.rename(columns={"category": "most_frequent_category"})[["customer_id", "most_frequent_category"]]

customers["tenure_days"] = (CUTOFF_DATE - customers["signup_date"]).dt.days

features_df = customers.merge(labels_df, on="customer_id")
features_df = features_df.merge(recency_df, on="customer_id", how="left")
features_df = features_df.merge(completed_metrics, on="customer_id", how="left")
features_df = features_df.merge(cancel_counts, on="customer_id", how="left")
features_df = features_df.merge(most_freq_cat, on="customer_id", how="left")

features_df["cancellation_rate"] = features_df["cancellation_rate"].fillna(0)
features_df["frequency"] = features_df["frequency"].fillna(0)
features_df["monetary"] = features_df["monetary"].fillna(0)

feature_cols = [
    "recency", "frequency", "monetary", "average_discount_pct", 
    "cancellation_rate", "average_delivery_minutes", 
    "most_frequent_category", "city", "age", "gender", "tenure_days"
]

X = features_df[feature_cols]
y = features_df["lapsed"]

# 3. Train-Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Check for Missing Values in Code
print("\n--- Missing Value Audit ---")
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
print(f"Missing values in X_train: {train_missing}")
print(f"Missing values in X_test: {test_missing}")
assert train_missing == 0 and test_missing == 0, "Error: Missing values detected!"
print("-> Confirmed: No feature column contains any missing values.")

# 5. Define Column Transformer with Pipeline
numeric_features = [
    "recency", "frequency", "monetary", "average_discount_pct", 
    "cancellation_rate", "average_delivery_minutes", "age", "tenure_days"
]
categorical_features = ["most_frequent_category", "city", "gender"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
    ]
)

# Fit preprocessor strictly on training data
preprocessor.fit(X_train)

# Transform train and test
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# 6. Compare Training Scaler Mean vs Full-Cohort Mean for Frequency
freq_idx = numeric_features.index("frequency")
train_freq_mean = preprocessor.named_transformers_["num"].mean_[freq_idx]
full_freq_mean = X["frequency"].mean()

print("\n--- Frequency Mean Comparison ---")
print(f"Fitted Training Scaler Frequency Mean: {train_freq_mean:.4f}")
print(f"Full-Cohort Frequency Mean: {full_freq_mean:.4f}")
print("-> Explanation: They differ slightly due to random sampling variance introduced by the stratified train-test split, as the training subset represents 80% of the full cohort.")
