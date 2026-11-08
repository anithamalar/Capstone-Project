import os
import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"
conn = sqlite3.connect(DB_PATH)

print("Running Q1 to Q7 SQL Analysis...")

# Helper function to read and execute blocks from queries.sql
with open("sql/queries.sql", "r") as f:
    sql_script = f.read()

# Split individual queries by semicolon
queries = [q.strip() for q in sql_script.split(";") if q.strip()]

for i, query in enumerate(queries, 1):
    print(f"\n--- Executing Q{i} ---")
    df = pd.read_sql_query(query, conn)
    print(f"Result shape: {df.shape}")
    print(df.head(3)) # Print top 3 rows as a sample

conn.close()
print("\n-> All SQL queries executed successfully!")
