import os
import sqlite3
import pandas as pd

DB_PATH = "data/instamart.db"
conn = sqlite3.connect(DB_PATH)

print("Running Q1 to Q7 SQL Analysis...\n")

with open("sql/queries.sql", "r") as f:
    sql_script = f.read()

# Split individual queries by semicolon
queries = [q.strip() for q in sql_script.split(";") if q.strip()]

for i, query in enumerate(queries, 1):
    print(f"==================== Q{i} RESULTS ====================")
    df = pd.read_sql_query(query, conn)
    print(f"Total Rows Returned: {len(df)}")
    print(df.head(5))
    print("=" * 55 + "\n")

conn.close()
print("-> All SQL queries executed and displayed successfully!")
