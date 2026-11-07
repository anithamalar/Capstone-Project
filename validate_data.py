import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"

print("Running validation script against SQLite database...")
conn = sqlite3.connect(DB_PATH)

# Load tables from database for checking
orders = pd.read_sql("SELECT * FROM orders", conn)
customers = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

# Assertions (Checks at least eight reference checkpoints)
# 1. Total order rows equals 12309
assert len(orders) == 12309, f"Expected 12309 order rows, got {len(orders)}"

# 2. Zero pending rows
pending_count = (orders["status"] == "pending").sum()
assert pending_count == 0, f"Expected 0 pending orders, got {pending_count}"

# 3. Exactly 1110 completed orders
completed_count = (orders["status"] == "completed").sum()
assert completed_count == 1110, f"Expected 1110 completed orders, got {completed_count}"

# 4. Exactly 1199 cancelled orders
cancelled_count = (orders["status"] == "cancelled").sum()
assert cancelled_count == 1199, f"Expected 1199 cancelled orders, got {cancelled_count}"

# 5. Zero missing order amounts
missing_amounts = orders["order_amount"].isnull().sum()
assert missing_amounts == 0, f"Expected 0 missing order amounts, got {missing_amounts}"

# 6. Completed orders amount sum equals INR 2,998,668.80 (allowing small floating point tolerance)
completed_sum = orders[orders["status"] == "completed"]["order_amount"].sum()
assert round(completed_sum, 2) == 2998668.80, f"Expected sum 2998668.80, got {completed_sum}"

# 7. Zero missing ages in customers table
missing_ages = customers["age"].isnull().sum()
assert missing_ages == 0, f"Expected 0 missing ages, got {missing_ages}"

# 8. All customer ages between 18 and 60
assert customers["age"].min() >= 18 and customers["age"].max() <= 60, "Customer ages out of expected 18-60 range"

print("ALL CHECKPOINTS PASSED")
