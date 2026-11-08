import sqlite3
import pandas as pd

print("SCRIPT HAS STARTED SUCCESSFULLY")

DB_PATH = "data/instamart.db"
conn = sqlite3.connect(DB_PATH)

# Let's run Q1 explicitly to test
q1 = """
SELECT c.customer_id, c.name, SUM(o.order_amount) AS total_spend
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.name
HAVING COUNT(o.order_id) > 500 AND SUM(o.order_amount) > 8000;
"""

print("Executing Q1...")
df_q1 = pd.read_sql_query(q1, conn)
print(f"Q1 Results Count: {len(df_q1)}")
print(df_q1.head())

conn.close()
print("SCRIPT FINISHED SUCCESSFULLY")
