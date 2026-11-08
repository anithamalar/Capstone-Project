import os
import pandas as pd

OUTPUT_DIR = os.path.join("data", "output")

print("Inspecting SQL Query Results from CSV files...\n")

for i in range(1, 8):
    file_path = os.path.join(OUTPUT_DIR, f"q{i}_result.csv")
    if os.path.exists(file_path):
        print(f"==================== Q{i} RESULTS ({file_path}) ====================")
        df = pd.read_csv(file_path)
        print(f"Total Rows: {len(df)}")
        print(df.head(5))
        print("\n")
    else:
        print(f"File for Q{i} not found at {file_path}")
