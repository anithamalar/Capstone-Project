import os
import re
import pandas as pd

# Define paths
RAW_DIR = "data/raw"
TICKETS_FILE = os.path.join(RAW_DIR, "tickets.csv")

print(f"Loading support tickets from {TICKETS_FILE}...")
tickets_df = pd.read_csv(TICKETS_FILE)

# Define exact regular expression patterns and their replacement placeholders
# Order matters: Email -> Phone -> Order ID
patterns = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
    ("ORDER_ID", re.compile(r"ORD-\d{6}"))
]

total_email_substitutions = 0
total_phone_substitutions = 0
total_order_substitutions = 0

redacted_texts = []

for idx, row in tickets_df.iterrows():
    text = row["text"]
    
    # Handle NaN or non-string inputs safely
    if pd.isna(text) or not isinstance(text, str):
        redacted_texts.append("")
        continue
        
    current_text = text
    
    # 1. Redact Emails
    current_text, email_count = patterns[0][1].subn("[EMAIL]", current_text)
    total_email_substitutions += email_count
    
    # 2. Redact Phones
    current_text, phone_count = patterns[1][1].subn("[PHONE]", current_text)
    total_phone_substitutions += phone_count
    
    # 3. Redact Order IDs
    current_text, order_count = patterns[2][1].subn("[ORDER_ID]", current_text)
    total_order_substitutions += order_count
    
    redacted_texts.append(current_text)

tickets_df["redacted_text"] = redacted_texts

total_substitutions = total_email_substitutions + total_phone_substitutions + total_order_substitutions

print("\n--- Task 2 PII Redaction Summary ---")
print(f"Emails redacted:   {total_email_substitutions} (Expected: 55)")
print(f"Phones redacted:   {total_phone_substitutions} (Expected: 91)")
print(f"Order IDs redacted: {total_order_substitutions} (Expected: 132)")
print(f"Total substitutions: {total_substitutions} (Expected: 278)")

# Verification check: re-applying patterns to redacted texts must find 0 matches
residual_matches = 0
for text in redacted_texts:
    for placeholder, pattern in patterns:
        matches = pattern.findall(text)
        if len(matches) > 0:
            residual_matches += len(matches)

print(f"\nResidual PII matches found after redaction: {residual_matches} (Expected: 0)")
assert residual_matches == 0, "Error: PII remains after redaction!"
print("-> Confirmed: Re-applying regex patterns finds nothing left.")
