import os
import pandas as pd

OUTPUT_DIR = os.path.join("data", "output")
print("Running diagnostic check on exported query outputs...\n")

expectations = {
    "q1_result.csv": (30, "Q1 (Completed orders > 500, spend > 8000)"),
    "q2_result.csv": (30, "Q2 (Revenue by city and category)"),
    "q3_result.csv": (11, "Q3 (Products above average price)"),
    "q4_result.csv": (25, "Q4 (Customers with no completed orders)"),
    "q5_result.csv": (1, "Q5 (Customer spend vs average summary row)"),
    "q6_result.csv": (None, "Q6 (Ranking window functions)"),
    "q7_result.csv": (183, "Q7 (Running total daily revenue)")
}

all_passed = True
for filename, (expected_rows, description) in expectations.items():
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ {description}: File not found!")
        all_passed = False
        continue
        
    df = pd.read_csv(path)
    actual_rows = len(df)
    
    if expected_rows is not None and actual_rows != expected_rows:
        print(f"❌ {description}: Expected {expected_rows} rows, but got {actual_rows} rows.")
        all_passed = False
    else:
        print(f"✓ {description}: Passed ({actual_rows} rows)")
        if filename == "q2_result.csv":
            total_rev = df["total_revenue"].sum()
            print(f"   -> Q2 Total Revenue: INR {total_rev:,.2f}")
        elif filename == "q5_result.csv":
            print(f"   -> Q5 Output Data:\n{df.to_string(index=False)}")
        elif filename == "q7_result.csv":
            if "running_total" in df.columns:
                final_total = df["running_total"].iloc[-1]
                print(f"   -> Q7 Final Running Total: INR {final_total:,.2f}")

if all_passed:
    print("\nAll exported query row counts and metrics match expectations perfectly!")
else:
    print("\nOne or more queries deviated from expected metrics. Let's inspect which one failed above.")
