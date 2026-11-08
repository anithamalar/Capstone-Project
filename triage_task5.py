import os
import re
import json
import pandas as pd
import mock_llm

RAW_DIR = "data/raw"
TICKETS_FILE = os.path.join(RAW_DIR, "tickets.csv")
ALLOWED_CATEGORIES = {"billing", "delivery", "app_issue", "refund"}

print(f"Loading support tickets from {TICKETS_FILE}...")
tickets_df = pd.read_csv(TICKETS_FILE)

# (Assuming guardrails and PII redaction from previous tasks are applied)
patterns = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
    ("ORDER_ID", re.compile(r"ORD-\d{6}"))
]

redacted_texts = []
for text in tickets_df["text"]:
    if pd.isna(text) or not isinstance(text, str):
        redacted_texts.append("")
        continue
    current_text = text
    current_text, _ = patterns[0][1].subn("[EMAIL]", current_text)
    current_text, _ = patterns[1][1].subn("[PHONE]", current_text)
    current_text, _ = patterns[2][1].subn("[ORDER_ID]", current_text)
    redacted_texts.append(current_text)

tickets_df["redacted_text"] = redacted_texts

with open("outputs/prompts.json", "r", encoding="utf-8") as f:
    prompts_data = json.load(f)
system_prompt = prompts_data["zero_shot"]

print("\n--- Running Task 5: Validation & Classification Pipeline ---")
mock_llm.reset()

results = []
status_counts = {"classified_json": 0, "classified_keyword": 0, "flagged_invalid_output": 0}

for idx, row in tickets_df.iterrows():
    ticket_id = row["ticket_id"]
    user_prompt = row["redacted_text"]
    
    if pd.isna(user_prompt) or str(user_prompt).strip() == "":
        continue
        
    try:
        response = mock_llm.chat(system_prompt, user_prompt)
    except Exception as e:
        response = str(e)
    
    status = "flagged_invalid_output"
    
    # 1. Try parsing as JSON
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "category" in parsed:
            cat = str(parsed["category"]).strip().lower()
            if cat in ALLOWED_CATEGORIES:
                status = "classified_json"
    except (json.JSONDecodeError, TypeError, ValueError):
        # 2. Not valid JSON, check via regex word extraction
        words = re.findall(r"[a-z]+", response.lower())
        matched_categories = [w for w in words if w in ALLOWED_CATEGORIES]
        
        if len(matched_categories) == 1:
            status = "classified_keyword"
            
    status_counts[status] += 1
    results.append({"ticket_id": ticket_id, "status": status, "response": response})

print("\n--- Task 5 Results Summary ---")
print(f"classified_json: {status_counts['classified_json']}")
print(f"classified_keyword: {status_counts['classified_keyword']}")
print(f"flagged_invalid_output: {status_counts['flagged_invalid_output']}")
