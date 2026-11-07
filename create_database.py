import os
import sqlite3
import pandas as pd

CLEAN_DIR = "data/clean"
DB_PATH = "data/instamart.db"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

print("Loading cleaned CSV files...")
customers = pd.read_csv(os.path.join(CLEAN_DIR, "customers.csv"))
products = pd.read_csv(os.path.join(CLEAN_DIR, "products.csv"))
orders = pd.read_csv(os.path.join(CLEAN_DIR, "orders.csv"))

# Connect to SQLite database (creates it if it doesn't exist)
print(f"Connecting to database at {DB_PATH}...")
conn = sqlite3.connect(DB_PATH)

# Write DataFrames to SQLite tables
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)

# Close connection
conn.close()
print("-> Successfully stored all tables into instamart.db!")
