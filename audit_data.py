import os
import pandas as pd

# Define path to raw data folder
RAW_DIR = os.path.join("data", "raw")

print("==================== DATA AUDIT REPORT ====================")

# 1. Load the raw tables (tickets.csv uses keep_default_na=False as required)
customers = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
products = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
tickets = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)

# Helper function to print shape, data types, and missing values
def audit_table(name, df):
    print(f"\n--- {name.upper()} ---")
    print(f"Shape: {df.shape}")
    print("\nData Types & Missing Values:")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    summary_df = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Count": missing_count,
        "Missing Percentage (%)": missing_pct.round(2)
    })
    print(summary_df)

audit_table("customers", customers)
audit_table("products", products)
audit_table("orders", orders)
audit_table("tickets", tickets)

print("\n==================== SPECIFIC AUDIT FIGURES ====================")

# Orders checks
total_orders = len(orders)
exact_dupes = orders.duplicated().sum()
pending_rows = (orders['status'] == 'pending').sum()
distinct_oids = orders['order_id'].nunique()

# Non-numeric amounts check
non_numeric_amounts = pd.to_numeric(orders['order_amount'], errors='coerce').isnull().sum()

# Delivery minutes checks
# Convert to numeric safely; non-numeric or empty strings become NaN
deliv_numeric = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
neg_deliv = (deliv_numeric < 0).sum()
blank_deliv = orders['delivery_minutes'].apply(lambda x: str(x).strip() == '').sum()

print(f"Orders rows: {total_orders}")
print(f"Exact duplicate rows: {exact_dupes}")
print(f"Pending rows: {pending_rows}")
print(f"Distinct order_id values: {distinct_oids}")
print(f"Non-numeric order_amounts: {non_numeric_amounts}")
print(f"Negative delivery_minutes: {neg_deliv}")
print(f"Blank delivery_minutes: {blank_deliv}")

# Customers checks
blank_ages = customers['age'].apply(lambda x: str(x).strip() == '').sum() + customers['age'].isnull().sum()
# Convert age to numeric for range checking, coercing blanks/errors to NaN
age_numeric = pd.to_numeric(customers['age'], errors='coerce')
ages_outside_range = ((age_numeric < 10) | (age_numeric > 100)).sum()
distinct_cities = customers['city'].nunique()
# Clean/non-blank gender spellings count
non_blank_genders = customers[customers['gender'].apply(lambda x: str(x).strip() != '')]['gender']
distinct_genders = non_blank_genders.nunique()
blank_genders = customers['gender'].apply(lambda x: str(x).strip() == '').sum() + customers['gender'].isnull().sum()

print(f"Blank ages: {blank_ages}")
print(f"Ages outside 10-100: {ages_outside_range}")
print(f"Distinct raw city spellings: {distinct_cities}")
print(f"Distinct non-blank raw gender spellings: {distinct_genders}")
print(f"Blank genders: {blank_genders}")
print("================================================================")
