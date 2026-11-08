import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 1. Load clean datasets & rebuild features
CLEAN_DIR = "data/clean"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading clean datasets and preparing features...")
orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))
customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

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

# Keep track of customer_ids aligned with features
X_with_id = features_df[["customer_id"] + feature_cols]
y_with_id = features_df[["customer_id", "lapsed"]]

# Train-Test Split with customer_id tracking
X_train, X_test, y_train, y_test = train_test_split(
    X_with_id, y, test_size=0.2, stratify=y, random_state=42
)

X_train_features = X_train.drop(columns=["customer_id"])
X_test_features = X_test.drop(columns=["customer_id"])

# 2. Setup Preprocessor Pipeline & Train Best Models
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

# Train Best Random Forest (max_depth=8, min_samples_leaf=5)
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42))
])
rf_pipeline.fit(X_train_features, y_train)

# Train Best Logistic Regression (C=1)
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=1, max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train_features, y_train)

# ==========================================
# (a) Model Interpretation
# ==========================================
print("\n--- (a) Model Interpretation ---")
# Get feature names after preprocessing
preprocessor.fit(X_train_features)
cat_encoder = preprocessor.named_transformers_["cat"]
encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
all_feature_names = numeric_features + encoded_cat_features

# Random Forest Feature Importances
rf_importances = rf_pipeline.named_steps["classifier"].feature_importances_
rf_imp_df = pd.DataFrame({"feature": all_feature_names, "importance": rf_importances})
rf_imp_sorted = rf_imp_df.sort_values(by="importance", ascending=False).head(5)

print("Top 5 Random Forest Features:")
for idx, row in rf_imp_sorted.reset_index(drop=True).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

print("\nBusiness Reading (Random Forest): Recency is overwhelmingly the dominant predictor, confirming that days elapsed since the last order is the primary indicator of customer lapse.")

# Logistic Regression Coefficients
lr_coefs = lr_pipeline.named_steps["classifier"].coef_[0]
lr_coef_df = pd.DataFrame({"feature": all_feature_names, "coefficient": lr_coefs})

print("\nLogistic Regression Feature Coefficients (Sample of 3):")
sample_features = ["recency", "frequency", "cancellation_rate"]
for feat in sample_features:
    coef_val = lr_coef_df[lr_coef_df["feature"] == feat]["coefficient"].values[0]
    sign = "Positive (+)" if coef_val > 0 else "Negative (-)"
    print(f"  - {feat}: {coef_val:.4f} ({sign})")

print("Business Reading (Logistic Regression): Recency has a positive coefficient (longer absence increases lapse risk), while frequency has a negative coefficient (higher past order counts protect against lapsing).")

# ==========================================
# (b) Persistence & Reproducibility Check
# ==========================================
print("\n--- (b) Model Persistence & Verification ---")
model_path = os.path.join(OUTPUT_DIR, "lapse_model.joblib")
joblib.dump(rf_pipeline, model_path)
print(f"Model saved to {model_path}")

# Reload model
loaded_model = joblib.load(model_path)

# Verify predictions match exactly
original_preds = rf_pipeline.predict(X_test_features)
reloaded_preds = loaded_model.predict(X_test_features)
preds_match = np.array_equal(original_preds, reloaded_preds)

print(f"Reloaded model predictions match original test predictions exactly: {preds_match}")

# ==========================================
# (c) Premium Customer Retention List
# ==========================================
print("\n--- (c) Premium Customer Retention List ---")
# Load segments table from Task 3 to get Premium flag
segments_path = os.path.join(CLEAN_DIR, "customer_segments.csv")
segments_df = pd.read_csv(segments_path)

# Filter test set customers who are in the Premium segment
test_customers_df = X_test.copy()
test_customers_df["predicted_lapse_prob"] = rf_pipeline.predict_proba(X_test_features)[:, 1]
test_customers_df["actually_lapsed"] = y_test.values

# Merge with segments dataframe to identify Premium status
test_merged = test_customers_df.merge(segments_df[["customer_id", "is_premium"]], on="customer_id", how="inner")
premium_test_customers = test_merged[test_merged["is_premium"] == True].copy()

# Rank by predicted lapse probability descending
premium_ranked = premium_test_customers.sort_values(by="predicted_lapse_prob", ascending=False)
top_retention_list = premium_ranked.head(10)[["customer_id", "predicted_lapse_prob", "actually_lapsed"]]
top_retention_list = top_retention_list.rename(columns={"predicted_lapse_prob": "lapse_probability"})

# Save retention list
retention_csv_path = os.path.join(OUTPUT_DIR, "retention_list.csv")
top_retention_list.to_csv(retention_csv_path, index=False)

truly_lapsed_count = top_retention_list["actually_lapsed"].sum()
print(f"Top retention list saved to {retention_csv_path} ({len(top_retention_list)} customers)")
print(f"Number of customers in top list who truly lapsed: {truly_lapsed_count} out of {len(top_retention_list)}")

print("\n--- Limitation Note ---")
print("Limitation Statement: The Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label; thus, this retention list is illustrative and not a leakage-free evaluation.")
