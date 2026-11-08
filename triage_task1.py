import os
import pandas as pd

# Define paths
RAW_DIR = "data/raw"
TICKETS_FILE = os.path.join(RAW_DIR, "tickets.csv")

# Constants
MAX_INPUT_TOKENS = 400

def est_tokens(text):
    """
    Estimates the token count of a given text string.
    Standard approximation uses whitespace-separated words.
    """
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(text.split())

print(f"Loading support tickets from {TICKETS_FILE}...")
tickets_df = pd.read_csv(TICKETS_FILE)
print(f"Total tickets loaded: {len(tickets_df)}")

# Process each ticket in the order they appear
guardrail_statuses = []
token_counts = []

for idx, row in tickets_df.iterrows():
    text = row["text"]
    
    # Check (a): Empty or whitespace-only text
    if pd.isna(text) or str(text).strip() == "":
        guardrail_statuses.append("skipped_empty")
        token_counts.append(0)
        continue
        
    # Estimate token count
    tokens = est_tokens(text)
    token_counts.append(tokens)
    
    # Check (b): Exceeds maximum token threshold
    if tokens > MAX_INPUT_TOKENS:
        guardrail_statuses.append("flagged_long")
        continue
        
    # Otherwise, passes input guardrails
    guardrail_statuses.append("valid")

tickets_df["guardrail_status"] = guardrail_statuses
tickets_df["estimated_tokens"] = token_counts

# Summary statistics
print("\n--- Task 1 Guardrail Summary ---")
status_counts = tickets_df["guardrail_status"].value_counts()
print(status_counts)

# Verify reference metrics
empty_count = (tickets_df["guardrail_status"] == "skipped_empty").sum()
long_count = (tickets_df["guardrail_status"] == "flagged_long").sum()
print(f"\nVerification -> Empty tickets: {empty_count} (Expected: 4)")
print(f"Verification -> Flagged long tickets: {long_count} (Expected: 4)")
