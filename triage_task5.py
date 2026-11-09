import os
import re
import time
import random
import json
import pandas as pd
import mock_llm

RAW_DIR = "data/raw"
TICKETS_FILE = os.path.join(RAW_DIR, "tickets.csv")
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
MAX_RETRIES = 3
BASE_DELAY = 0.05

print(f"Loading support tickets from {TICKETS_FILE}...")
tickets_df = pd.read_csv(TICKETS_FILE)

# Task 1 Guardrails
MAX_INPUT_TOKENS = 400
def est_tokens(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(text.split())

guardrail_statuses = []
for text in tickets_df["text"]:
    if pd.isna(text) or str(text).strip() == "":
        guardrail_statuses.append("skipped_empty")
    elif est_tokens(text) > MAX_INPUT_TOKENS:
        guardrail_statuses.append("flagged_long")
    else:
        guardrail_statuses.append("valid")

tickets_df["guardrail_status"] = guardrail_statuses

# Task 2 PII Redaction
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

# Resilient LLM Call with Retry & Backoff for Task 5
def call_llm_with_retry(ticket_id, messages):
    for n in range(MAX_RETRIES + 1):
        try:
            response_obj = mock_llm.chat(messages)
            return response_obj.choices[0].message.content
        except Exception as e:
            if n == MAX_RETRIES:
                raise e
            wait_time = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            time.sleep(wait_time)

print("\n--- Running Task 5: Validation & Classification Pipeline ---")
mock_llm.reset()

valid_tickets = tickets_df[tickets_df["guardrail_status"] == "valid"]
status_counts = {"classified_json": 0, "classified_keyword": 0, "flagged_invalid_output": 0}
results = []

for idx, row in valid_tickets.iterrows():
    ticket_id = row["ticket_id"]
    user_prompt = row["redacted_text"]
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response_text = call_llm_with_retry(ticket_id, messages)
    except Exception as e:
        response_text = str(e)
    
    status = "flagged_invalid_output"
    
    # 1. Parse as JSON and check if category is one of the allowed values
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict) and "category" in parsed:
            cat = str(parsed["category"]).strip().lower()
            if cat in ALLOWED_CATEGORIES:
                status = "classified_json"
    except (json.JSONDecodeError, TypeError, ValueError):
        # 2. Fallback: Extract lowercase words using re.findall
        words = re.findall(r"[a-z]+", response_text.lower())
        matched_categories = [w for w in words if w in ALLOWED_CATEGORIES]
        
        if len(matched_categories) == 1:
            status = "classified_keyword"
            
    status_counts[status] += 1
    results.append({"ticket_id": ticket_id, "status": status, "response": response_text})

print("\n--- Task 5 Results Summary ---")
print(f"classified_json: {status_counts['classified_json']}")
print(f"classified_keyword: {status_counts['classified_keyword']}")
print(f"flagged_invalid_output: {status_counts['flagged_invalid_output']}")
