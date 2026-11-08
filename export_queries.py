import os
import sqlite3
import pandas as pd

# Define paths
DB_PATH = "data/instamart.db"
OUTPUT_DIR = os.path.join("data", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Connecting to database and exporting query results...")
conn = sqlite3.connect(DB_PATH)

# Read queries from sql/queries.sql
with open("sql/queries.sql", "r") as f:
    sql_script = f.read()

# Split individual queries by semicolon
queries = [q.strip() for q in sql_script.split(";") if q.strip()]

for i, query in enumerate(queries, 1):
    df = pd.read_sql_query(query, conn)
    output_file = os.path.join(OUTPUT_DIR, f"q{i}_result.csv")
    df.to_csv(output_file, index=False)
    print(f"-> Exported Q{i} to {output_file} ({len(df)} rows)")

conn.close()
print("\nAll query outputs successfully exported to CSV!")
