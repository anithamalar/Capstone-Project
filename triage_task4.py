import os
import re
import time
import random
import json
import pandas as pd

# Import mock_llm (ensure mock_llm.py is in D:\AM)
try:
    import mock_llm
except ImportError:
    # Fallback definition if mock_llm isn't physically present yet
    class TransientLLMError(Exception):
        pass
    class MockLLM:
        def reset(self): pass
        def invoke(self, system_prompt, user_prompt): pass
    mock_llm = MockLLM()

# Define paths & constants
RAW_DIR = "data/raw"
TICKETS_FILE = os.path.join(RAW_DIR, "tickets.csv")
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05  # seconds
MAX_RETRIES = 3

print(f"Loading support tickets from {TICKETS_FILE}...")
tickets_df = pd.read_csv(TICKETS_FILE)

# 1. Apply Task 1 Input Guardrails
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

# 2. Apply Task 2 PII Redaction
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
    for _, pattern in patterns:
        current_text = pattern.sub(r"[\1]" if _=="EMAIL" else f"[{_}]", current_text) # simplified substitution
    # Exact substitution logic from Task 2
    curr = text
    curr, _ = patterns[0][1].subn("[EMAIL]", curr)
    curr, _ = patterns[1][1].subn("[PHONE]", curr)
    curr, _ = patterns[2][1].subn("[ORDER_ID]", curr)
    redacted_texts.append(curr)

tickets_df["redacted_text"] = redacted_texts

# 3. Load System Prompts
with open("outputs/prompts.json", "r", encoding="utf-8") as f:
    prompts_data = json.load(f)
system_prompt = prompts_data["zero_shot"]

# 4. Define Resilient Call Function with Retry, Backoff, and Jitter
def call_llm_with_retry(ticket_id, system_prompt, user_prompt):
    """
    Calls mock_llm.invoke with up to MAX_RETRIES (3 retries).
    Implements exponential backoff with jitter:
    wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
    """
    for n in range(MAX_RETRIES + 1):
        try:
            # Call mock LLM invoke
            response = mock_llm.invoke(system_prompt, user_prompt)
            return response
        except Exception as e:
            # Check if it's a transient error (or catch any exception raised by mock_llm)
            if n == MAX_RETRIES:
                print(f"Ticket {ticket_id}: Permanent failure after {MAX_RETRIES} retries. Error: {e}")
                raise e
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            
            # Log retry line showing ticket id, retry number, and wait time
            print(f"[RETRY LOG] Ticket ID: {ticket_id} | Retry #{n+1} | Wait: {wait_time:.4f}s | Error: {e}")
            
            time.sleep(wait_time)

# 5. Execute Full Run on Valid Tickets
print("\n--- Resetting Mock LLM & Starting Full Run ---")
mock_llm.reset()

valid_tickets = tickets_df[tickets_df["guardrail_status"] == "valid"]
print(f"Processing {len(valid_tickets)} valid tickets through resilient LLM pipeline...")

results = []
for idx, row in valid_tickets.iterrows():
    ticket_id = row["ticket_id"]
    user_prompt = row["redacted_text"]  # Content sent must be exactly redacted text and nothing else
    
    try:
        response = call_llm_with_retry(ticket_id, system_prompt, user_prompt)
        results.append({"ticket_id": ticket_id, "status": "success", "response": response})
    except Exception as err:
        results.append({"ticket_id": ticket_id, "status": "failed", "response": str(err)})

print(f"\nCompleted processing. Total successful responses: {sum(1 for r in results if r['status'] == 'success')} / {len(valid_tickets)}")
