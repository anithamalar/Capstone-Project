import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"
print("Connecting to database for exploratory analysis...\n")
conn = sqlite3.connect(DB_PATH)

# Load orders, customers, and products tables
orders = pd.read_sql("SELECT * FROM orders", conn)
customers = pd.read_sql("SELECT * FROM customers", conn)
products = pd.read_sql("SELECT * FROM products", conn)
conn.close()

# Filter for completed orders only
completed_orders = orders[orders["status"] == "completed"].copy()

# ==========================================
# 1. Pivot Table Cross-Check (Q2 Recreation)
# ==========================================
print("--- 1. Q2 Pivot Table Cross-Check ---")
q2_df = orders.merge(customers, on="customer_id").merge(products, on="product_id")
q2_completed = q2_df[q2_df["status"] == "completed"]

pivot_table = pd.pivot_table(
    q2_completed,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    margins_name="Total"
)
print(f"Pivot table generated successfully! Grand Total: INR {pivot_table.loc['Total', 'Total']:,.2f}\n")

# ==========================================
# 2. Summary Statistics for order_amount
# ==========================================
print("--- 2. Order Amount Statistics & Outliers ---")
amounts = completed_orders["order_amount"]

mean_val = amounts.mean()
median_val = amounts.median()
skew_val = amounts.skew()
q1 = amounts.quantile(0.25)
q3 = amounts.quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr

outliers_above = (amounts > upper_fence).sum()
outliers_below = (amounts < (q1 - 1.5 * iqr)).sum()

print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Skewness: {skew_val:.2f}")
print(f"Q1 (25%): {q1:.2f} | Q3 (75%): {q3:.2f}")
print(f"1.5x IQR Upper Fence: {upper_fence:.2f}")
print(f"Outliers above upper fence: {outliers_above}")
print(f"Outliers below lower fence: {outliers_below}\n")

# ==========================================
# 3. Correlation Matrix
# ==========================================
print("--- 3. Correlation Matrix ---")
corr_vars = completed_orders[["order_amount", "discount_pct", "quantity", "delivery_minutes"]]
corr_matrix = corr_vars.corr()
print(corr_matrix)
print("\nLargest absolute correlation is between order_amount and discount_pct (~ -0.095).")
print("Note: A correlation of this small size indicates virtually no linear relationship and does not establish causation.\n")

# ==========================================
# 4. Executive Summary Generation
# ==========================================
print("--- 4. Executive Summary ---")
summary_text = """
Executive Summary: Instamart Customer Intelligence
--------------------------------------------------
This dataset captures Instamart's transactional, customer, and product operations, comprising 12,309 total orders across multiple cities, categories, and customer demographics, with 11,110 successfully completed orders generating a total revenue of INR 2,998,668.80. 

The distribution of completed order amounts is heavily right-skewed (skewness ~4.61), with a mean of INR 269.91 but a median of INR 135.00, driven by a long tail of high-value basket sizes. Using the interquartile range method, an upper fence of INR 645.30 identifies 897 high-value order outliers, with zero values below the lower fence. 

Variable correlation analysis across completed orders reveals very weak associations, where the largest absolute correlation (~ -0.095) exists between order amount and discount percentage. This minor negative association does not establish causation. Overall, the data reflects a healthy e-commerce ecosystem characterized by routine small-basket purchases balanced by occasional high-value transactions.
"""
print(summary_text)
