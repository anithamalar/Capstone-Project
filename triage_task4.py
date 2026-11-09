import os
import re
import time
import random
import json
import pandas as pd
import mock_llm

RAW_DIR = "data/raw"
TICKETS_FILE = os.path.join(RAW_DIR, "tickets.csv")
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

print(f"Loading support tickets from {TICKETS_FILE}...")
tickets_df = pd.read_csv(TICKETS_FILE)

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

def call_llm_with_retry(ticket_id, system_prompt, user_prompt):
    for n in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(system_prompt, user_prompt)
            return response
        except Exception as e:
            if n == MAX_RETRIES:
                print(f"Ticket {ticket_id}: Permanent failure after {MAX_RETRIES} retries. Error: {e}")
                raise e
            wait_time = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"[RETRY LOG] Ticket ID: {ticket_id} | Retry #{n+1} | Wait: {wait_time:.4f}s | Error: {e}")
            time.sleep(wait_time)

print("\n--- Resetting Mock LLM & Starting Full Run ---")
mock_llm.reset()

valid_tickets = tickets_df[tickets_df["guardrail_status"] == "valid"]
print(f"Processing {len(valid_tickets)} valid tickets through resilient LLM pipeline...")

results = []
for idx, row in valid_tickets.iterrows():
    ticket_id = row["ticket_id"]
    user_prompt = row["redacted_text"]
    try:
        response = call_llm_with_retry(ticket_id, system_prompt, user_prompt)
        results.append({"ticket_id": ticket_id, "status": "success", "response": response})
    except Exception as err:
        results.append({"ticket_id": ticket_id, "status": "failed", "response": str(err)})

success_count = sum(1 for r in results if r["status"] == "success")
print(f"\nCompleted processing. Total successful responses: {success_count} / {len(valid_tickets)}")
