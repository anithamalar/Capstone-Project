import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"

print("Running validation script against SQLite database...")
conn = sqlite3.connect(DB_PATH)
orders = pd.read_sql("SELECT * FROM orders", conn)
customers = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

# Asserting at least eight reference checkpoints from Tasks 3 and 4
assert len(orders) == 12309
print("✓ Checkpoint 1 Passed: Total orders = 12,309")

assert (orders["status"] == "pending").sum() == 0
print("✓ Checkpoint 2 Passed: Pending orders = 0")

assert (orders["status"] == "completed").sum() == 1110
print("✓ Checkpoint 3 Passed: Completed orders = 1,110")

assert (orders["status"] == "cancelled").sum() == 1199
print("✓ Checkpoint 4 Passed: Cancelled orders = 1,199")

assert orders["order_amount"].isnull().sum() == 0
print("✓ Checkpoint 5 Passed: Missing order amounts = 0")

assert round(orders[orders["status"] == "completed"]["order_amount"].sum(), 2) == 2998668.80
print("✓ Checkpoint 6 Passed: Completed revenue sum matches INR 2,998,668.80")

assert customers["age"].isnull().sum() == 0
print("✓ Checkpoint 7 Passed: Missing customer ages = 0")

assert customers["age"].min() >= 18 and customers["age"].max() <= 60
print("✓ Checkpoint 8 Passed: Customer ages bounded between 18 and 60")

print("\nALL CHECKPOINTS PASSED")
