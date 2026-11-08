import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load clean datasets
CLEAN_DIR = "data/clean"
print("Loading orders and customers datasets...")
orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))

# Convert date columns to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# 2. Define Cutoff and Window
CUTOFF_DATE = pd.Timestamp("2025-09-01 00:00:00")
LAPSE_WINDOW_END = pd.Timestamp("2025-10-01 00:00:00")

# 3. Define Cohort: Customers with at least one completed order BEFORE the cutoff
orders_before_cutoff = orders[orders["order_ts"] < CUTOFF_DATE].copy()
completed_before = orders_before_cutoff[orders_before_cutoff["status"] == "completed"]

cohort_customer_ids = completed_before["customer_id"].unique()
print(f"Total customers in cohort (at least 1 completed order before cutoff): {len(cohort_customer_ids)}")

# 4. Define Label (lapsed = 1 if NO completed order between 2025-09-01 and 2025-09-30)
# Orders in the observation window (Sept 2025)
sept_orders = orders[
    (orders["order_ts"] >= CUTOFF_DATE) & 
    (orders["order_ts"] < LAPSE_WINDOW_END) & 
    (orders["status"] == "completed")
]
active_in_sept = sept_orders["customer_id"].unique()

# Build label dataframe
labels_df = pd.DataFrame({"customer_id": cohort_customer_ids})
labels_df["lapsed"] = labels_df["customer_id"].apply(lambda cid: 0 if cid in active_in_sept else 1)

lapsed_count = labels_df["lapsed"].sum()
total_cohort = len(labels_df)
print(f"Lapsed customers: {lapsed_count} ({lapsed_count / total_cohort * 100:.2f}%)")

# 5. Feature Engineering (using ONLY data before cutoff)
# Merge orders before cutoff with products to get category
orders_prod = orders_before_cutoff.merge(products, on="product_id", how="left")

# Recency: days from cutoff to last completed order before cutoff
recency_df = completed_before.groupby("customer_id")["order_ts"].max().reset_index()
recency_df["recency"] = (CUTOFF_DATE - recency_df["order_ts"]).dt.days
recency_df = recency_df[["customer_id", "recency"]]

# Frequency & Monetary & Avg Discount & Avg Delivery for completed orders
completed_metrics = completed_before.groupby("customer_id").agg(
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    average_discount_pct=("discount_pct", "mean"),
    average_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

# Cancellation rate: cancelled orders ÷ all orders with a final status before cutoff
all_final_orders = orders_before_cutoff[orders_before_cutoff["status"].isin(["completed", "cancelled"])]
cancel_counts = all_final_orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    cancelled_orders=("status", lambda s: (s == "cancelled").sum())
).reset_index()
cancel_counts["cancellation_rate"] = cancel_counts["cancelled_orders"] / cancel_counts["total_orders"]
cancel_counts = cancel_counts[["customer_id", "cancellation_rate"]]

# Most frequent product category
cat_counts = orders_prod.groupby(["customer_id", "category"]).size().reset_index(name="count")
most_freq_cat = cat_counts.sort_values(["customer_id", "count"], ascending=[True, False]).drop_duplicates("customer_id")
most_freq_cat = most_freq_cat.rename(columns={"category": "most_frequent_category"})[["customer_id", "most_frequent_category"]]

# Tenure in days: cutoff minus signup_date
customers["tenure_days"] = (CUTOFF_DATE - customers["signup_date"]).dt.days

# Combine all features with customer profile table
features_df = customers.merge(labels_df, on="customer_id")
features_df = features_df.merge(recency_df, on="customer_id", how="left")
features_df = features_df.merge(completed_metrics, on="customer_id", how="left")
features_df = features_df.merge(cancel_counts, on="customer_id", how="left")
features_df = features_df.merge(most_freq_cat, on="customer_id", how="left")

# Fill missing values for customers with no cancellations or specific metrics if any
features_df["cancellation_rate"] = features_df["cancellation_rate"].fillna(0)
features_df["frequency"] = features_df["frequency"].fillna(0)
features_df["monetary"] = features_df["monetary"].fillna(0)

# Select final feature columns and label
feature_cols = [
    "recency", "frequency", "monetary", "average_discount_pct", 
    "cancellation_rate", "average_delivery_minutes", 
    "most_frequent_category", "city", "age", "gender", "tenure_days"
]

X = features_df[feature_cols]
y = features_df["lapsed"]

# 6. Train-Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining rows: {len(X_train)} | Test rows: {len(X_test)}")

# 7. Baseline Evaluation: Always predicting "not lapsed" (0) on the test set
y_pred_baseline = [0] * len(y_test)
base_accuracy = accuracy_score(y_test, y_pred_baseline)
base_f1 = f1_score(y_test, y_pred_baseline, pos_label=1, zero_division=0)

print(f"\n--- Baseline Model Performance (Always 'Not Lapsed') ---")
print(f"Accuracy: {base_accuracy * 100:.1f}%")
print(f"Lapsed-Class F1 Score: {base_f1:.4f}")
