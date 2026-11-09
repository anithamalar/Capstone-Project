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

def run_pipeline(prompt_key):
    system_prompt = prompts_data[prompt_key]
    mock_llm.reset()
    
    valid_tickets = tickets_df[tickets_df["guardrail_status"] == "valid"]
    classified_count = 0
    correct_count = 0
    total_tokens = 0
    
    # For confusion matrix: true vs predicted over classified tickets
    confusion = {t: {p: 0 for p in ALLOWED_CATEGORIES} for t in ALLOWED_CATEGORIES}
    
    for idx, row in valid_tickets.iterrows():
        user_prompt = row["redacted_text"]
        true_cat = str(row.get("true_category", "")).strip().lower()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Retry logic
        response_text = ""
        resp_obj = None
        for n in range(MAX_RETRIES + 1):
            try:
                resp_obj = mock_llm.chat(messages)
                response_text = resp_obj.choices[0].message.content
                break
            except Exception as e:
                if n == MAX_RETRIES:
                    response_text = str(e)
                else:
                    time.sleep(BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15))
        
        if resp_obj and hasattr(resp_obj, "usage"):
            total_tokens += resp_obj.usage.total_tokens
            
        status = "flagged_invalid_output"
        pred_cat = None
        
        # Task 5 Validation
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "category" in parsed:
                cat = str(parsed["category"]).strip().lower()
                if cat in ALLOWED_CATEGORIES:
                    status = "classified_json"
                    pred_cat = cat
        except (json.JSONDecodeError, TypeError, ValueError):
            words = re.findall(r"[a-z]+", response_text.lower())
            matched_categories = [w for w in words if w in ALLOWED_CATEGORIES]
            if len(matched_categories) == 1:
                status = "classified_keyword"
                pred_cat = matched_categories[0]
                
        if status in {"classified_json", "classified_keyword"} and pred_cat:
            classified_count += 1
            if true_cat in ALLOWED_CATEGORIES:
                confusion[true_cat][pred_cat] += 1
                if pred_cat == true_cat:
                    correct_count += 1
                    
    accuracy = (correct_count / classified_count * 100) if classified_count > 0 else 0
    tokens_per_ticket = (total_tokens / classified_count) if classified_count > 0 else 0
    
    return {
        "classified": classified_count,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_ticket": tokens_per_ticket,
        "confusion": confusion
    }

print("\n--- Running Zero-Shot Pipeline ---")
zs_results = run_pipeline("zero_shot")

print("\n--- Running Few-Shot Pipeline ---")
fs_results = run_pipeline("few_shot")

print("\n================ REPORT ================")
print(f"Zero-Shot -> Classified: {zs_results['classified']}, Accuracy: {zs_results['accuracy']:.2f}%, Total Tokens: {zs_results['total_tokens']}, Tokens/Ticket: {zs_results['tokens_per_ticket']:.2f}")
print(f"Few-Shot  -> Classified: {fs_results['classified']}, Accuracy: {fs_results['accuracy']:.2f}%, Total Tokens: {fs_results['total_tokens']}, Tokens/Ticket: {fs_results['tokens_per_ticket']:.2f}")

print("\n--- Zero-Shot Confusion Matrix (True vs Predicted) ---")
print(f"{'True \\ Pred':<12} | " + " | ".join([f"{c:>10}" for c in ALLOWED_CATEGORIES]))
print("-" * 60)
for t_cat in ALLOWED_CATEGORIES:
    row_str = f"{t_cat:<12} | " + " | ".join([f"{zs_results['confusion'][t_cat][p_cat]:10d}" for p_cat in ALLOWED_CATEGORIES])
    print(row_str)
