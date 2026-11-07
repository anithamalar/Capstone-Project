import os
import numpy as np
import pandas as pd

# Define paths
RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)

print("Loading raw customers data...")
cust = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))

# Track original rows for logging/documentation
initial_rows = len(cust)

# 1. City Standardisation
# Remove leading/trailing spaces and apply title case
original_cities = cust["city"].copy()
cust["city"] = cust["city"].str.strip().str.title()
cities_modified_count = (original_cities != cust["city"]).sum()
print(f"-> City standardisation modified {cities_modified_count} rows.")

# 2. Gender Mapping
# Map raw gender values to 'male', 'female', or 'unknown' (blank -> unknown)
def clean_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g_str = str(g).strip().lower()
    if g_str in ["m", "male"]:
        return "male"
    elif g_str in ["f", "female"]:
        return "female"
    return "unknown"

cust["gender"] = cust["gender"].apply(clean_gender)

# 3. Age Cleaning & Signup Date Conversion
# Treat ages outside 10-100 as missing (NaN)
cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan

# Fill missing ages using the median age of each customer's respective city via groupby and transform
city_median_ages = cust.groupby("city")["age"].transform("median")
cust["age"] = cust["age"].fillna(city_median_ages)

# Convert signup_date to a datetime type
cust["signup_date"] = pd.to_datetime(cust["signup_date"])

# 4. Save the cleaned file
output_path = os.path.join(CLEAN_DIR, "customers.csv")
cust.to_csv(output_path, index=False)
print(f"-> Cleaned customers saved to {output_path}")

print("\n==================== CHECKPOINT VERIFICATION ====================")
print(f"Missing ages count: {cust['age'].isnull().sum()} (Expected: 0)")
print(f"Minimum age: {cust['age'].min()}, Maximum age: {cust['age'].max()} (Expected: between 18 and 60)")
print("\nCity distribution after cleaning:")
print(cust["city"].value_counts())
print("\nGender distribution after cleaning:")
print(cust["gender"].value_counts())
print("=================================================================")
