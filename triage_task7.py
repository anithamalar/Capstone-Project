import os
import re
import time
import random
import json
import pandas as pd
import mock_llm

RAW_DIR = "data/raw"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TICKETS_FILE = os.path.join(RAW_DIR, "tickets.csv")
SEGMENTS_FILE = os.path.join(OUTPUTS_DIR, "customer_segments.csv")
PROMPTS_FILE = os.path.join(OUTPUTS_DIR, "prompts.json")

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
MAX_RETRIES = 3
BASE_DELAY = 0.05

print(f"Loading support tickets from {TICKETS_FILE}...")
tickets_df = pd.read_csv(TICKETS_FILE)

# 1. Task 1 Guardrails
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

# 2. Task 2 PII Redaction
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

with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    prompts_data = json.load(f)

# Load Customer Segments if available
segments_dict = {}
if os.path.exists(SEGMENTS_FILE):
    seg_df = pd.read_csv(SEGMENTS_FILE)
    for _, row in seg_df.iterrows():
        cid = str(row.get("customer_id", ""))
        segments_dict[cid] = {
            "is_premium": bool(row.get("is_premium", False)),
            "segment_name": row.get("segment_name", "")
        }

def run_pipeline_with_budget(prompt_key, token_budget):
    system_prompt = prompts_data[prompt_key]
    mock_llm.reset()
    
    running_tokens = 0
    results = []
    exhausted_count = 0
    
    for idx, row in tickets_df.iterrows():
        ticket_id = row["ticket_id"]
        customer_id = row.get("customer_id", "")
        g_status = row["guardrail_status"]
        
        record = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": g_status,
            "category": "",
            "segment_name": "",
            "priority": "",
            "route": "",
            "tokens_used": 0
        }
        
        if g_status != "valid":
            results.append(record)
            continue
            
        # Check token budget BEFORE model call
        if running_tokens >= token_budget:
            record["status"] = "budget_exhausted"
            exhausted_count += 1
            results.append(record)
            continue
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["redacted_text"]}
        ]
        
        response_text = ""
        resp_obj = None
        call_tokens = 0
        
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
            call_tokens = resp_obj.usage.total_tokens
            
        running_tokens += call_tokens
        record["tokens_used"] = call_tokens
        
        status = "flagged_invalid_output"
        pred_cat = ""
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
                
        record["status"] = status
        record["category"] = pred_cat
        results.append(record)
        
    return pd.DataFrame(results), exhausted_count

# (a) Run zero-shot with budget demo (Budget = 6000)
demo_budget = 6000
print(f"\n--- Running Task 7(a): Zero-Shot Budget Demo (Budget={demo_budget}) ---")
demo_df, num_exhausted = run_pipeline_with_budget("zero_shot", demo_budget)

if num_exhausted == 0:
    demo_budget = 5000
    print(f"No tickets exhausted at 6000. Retrying with budget = {demo_budget}...")
    demo_df, num_exhausted = run_pipeline_with_budget("zero_shot", demo_budget)

demo_output_path = os.path.join(OUTPUTS_DIR, "triage_budget_demo.csv")
demo_df.to_csv(demo_output_path, index=False)
print(f"Saved budget demo results to {demo_output_path}. Tickets budget_exhausted: {num_exhausted}")

# (b) Run few-shot with TOKEN_BUDGET = 100000 and apply routing rules
print(f"\n--- Running Task 7(b): Few-Shot Final Routing (Budget=100000) ---")
final_df, _ = run_pipeline_with_budget("few_shot", 100000)

routing_results = []
for _, row in final_df.iterrows():
    status = row["status"]
    cat = row["category"]
    cid = str(row["customer_id"])
    
    seg_info = segments_dict.get(cid, {"is_premium": False, "segment_name": ""})
    is_premium = seg_info["is_premium"]
    segment_name = seg_info["segment_name"] if status in {"classified_json", "classified_keyword"} else ""
    
    if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
        route = "human_review"
        priority = "NORMAL"
        cat = ""
        segment_name = ""
    elif cat == "fraud":
        route = "security_team"
        priority = "HIGH"
    elif is_premium:
        route = "premium_desk"
        priority = "HIGH"
    else:
        route = "support_queue"
        priority = "NORMAL"
        
    routing_results.append({
        "ticket_id": row["ticket_id"],
        "customer_id": row["customer_id"],
        "status": status,
        "category": cat if status in {"classified_json", "classified_keyword"} else "",
        "segment_name": segment_name,
        "priority": priority,
        "route": route,
        "tokens_used": row["tokens_used"]
    })

final_output_df = pd.DataFrame(routing_results)
final_output_path = os.path.join(OUTPUTS_DIR, "triage_results.csv")
final_output_df.to_csv(final_output_path, index=False)
print(f"Saved final triage results to {final_output_path}")
