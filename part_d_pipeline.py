import os
import glob
import json
import sqlite3
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICIES_DIR = "data/policies"
DB_PATH = "data/instamart.db"
SCENARIOS_FILE = "data/raw/agent_scenarios.csv"
OUTPUTS_DIR = "outputs"
STATE_DIR = "state"
LEDGER_PATH = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")
RUNS_JSONL_PATH = os.path.join(OUTPUTS_DIR, "agent_runs.jsonl")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# ==========================================
# 1. CORPUS CHUNKING & TF-IDF VECTORIZER (Tasks 1-3)
# ==========================================
def build_corpus(chunk_size=40, overlap=10):
    all_chunks = []
    policy_files = glob.glob(os.path.join(POLICIES_DIR, "*.txt"))
    for file_path in sorted(policy_files):
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        words = content.split()
        chunks = []
        step = chunk_size - overlap
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start += step
        for idx, chunk in enumerate(chunks):
            all_chunks.append({"source": filename, "text": chunk})
    return all_chunks

corpus_chunks = build_corpus()
corpus_texts = [c["text"] for c in corpus_chunks]
vectorizer = TfidfVectorizer()
corpus_vectors = vectorizer.fit_transform(corpus_texts)

def search_policy(query):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, corpus_vectors).flatten()
    top_indices = np.argsort(sims)[::-1][:3]
    results = []
    for idx in top_indices:
        results.append({
            "source": corpus_chunks[idx]["source"],
            "score": float(sims[idx]),
            "text": corpus_chunks[idx]["text"]
        })
    return {"results": results}

# ==========================================
# 2. TOOL IMPLEMENTATIONS (Task 5)
# ==========================================
def get_order(order_id):
    if not os.path.exists(DB_PATH):
        return {"error": "database not found"}
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", 
            (order_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "order_id": row[0],
                "status": row[1],
                "order_amount": float(row[2]),
                "delivery_minutes": int(row[3])
            }
        else:
            return {"error": "order not found"}
    except Exception as e:
        return {"error": str(e)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0.0
    return {"refund_inr": refund}

def append_to_ledger(tool_name, order_id, refund_inr):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def issue_refund(order_id, refund_inr):
    append_to_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    append_to_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order details including status, order amount, and delivery minutes from the database.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents corpus for information relevant to the query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    },
    {
        "name": "calculate_refund",
        "description": "Calculate the eligible refund amount based on order amount and delivery duration.",
        "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "integer"}}, "required": ["order_amount", "delivery_minutes"]}
    },
    {
        "name": "issue_refund",
        "description": "Issue the refund automatically for approved amounts up to INR 1000 and log it to the ledger.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate refund requests exceeding INR 1000 to a human support lead and log it to the ledger.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}
    }
]

# ==========================================
# 3. SAFE REACT AGENT RUNNER (Tasks 6-8)
# ==========================================
def run_agent_master(user_message, scenario_id, run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    start_time = time.monotonic()
    system_content = (
        "You are an AGENTic policy and refund assistant. Use the provided tools to lookup orders, "
        "search policies, calculate refunds, and either issue refunds or escalate to humans. "
        "Begin your final answer with one of the standard status words: REFUND, ESCALATE, DENIED, NOT_FOUND."
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]
    
    total_tokens = 0
    llm_calls = 0
    steps = []
    executed_tools = []
    step_count = 0
    
    for iteration in range(max_iterations):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: Time limit exceeded",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": total_tokens,
                "executed_tools": executed_tools
            }
            
        mock_llm.reset()
        response_obj = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        
        if hasattr(response_obj, "usage") and response_obj.usage:
            total_tokens += response_obj.usage.total_tokens
            
        if total_tokens > token_limit:
            return {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: Token limit exceeded",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": total_tokens,
                "executed_tools": executed_tools
            }
            
        msg = response_obj.choices[0].message
        
        if not getattr(msg, "tool_calls", None):
            final_ans = msg.content if msg.content else ""
            status = "UNKNOWN"
            if ":" in final_ans:
                status = final_ans.split(":")[0].strip().upper()
            return {
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_ans,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": total_tokens,
                "executed_tools": executed_tools
            }
            
        tool_calls_list = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls_list})
        
        for tc in msg.tool_calls:
            t_name = tc.name
            t_args = tc.arguments
            
            if t_name in TOOL_MAP:
                try:
                    obs = TOOL_MAP[t_name](**t_args)
                except Exception as e:
                    obs = {"error": str(e)}
            else:
                obs = {"error": f"Unknown tool {t_name}"}
                
            step_count += 1
            executed_tools.append(t_name)
            steps.append({"step": step_count, "tool": t_name, "arguments": t_args, "observation": obs})
            
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": t_name, "content": json.dumps(obs)})
            
            # Save Checkpoint
            checkpoint_data = {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "messages": messages,
                "tokens_used": total_tokens,
                "llm_calls": llm_calls,
                "step_number": step_count,
                "executed_tools": executed_tools,
                "steps": steps
            }
            with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
                
    return {
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: Max iterations reached",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": total_tokens,
        "executed_tools": executed_tools
    }

# ==========================================
# 4. EXECUTE ALL SCENARIOS & GENERATE REPORT
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("Starting Part D Master Pipeline Execution")
    print("==========================================")
    
    if os.path.exists(RUNS_JSONL_PATH):
        os.remove(RUNS_JSONL_PATH)
        
    scenarios_df = pd.read_csv(SCENARIOS_FILE)
    summary_rows = []
    
    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        
        res = run_agent_master(msg, sid, run_id=f"run_{sid.lower()}")
        
        # Extract refund amount from final answer if present
        refund_amt = "N/A"
        if "REFUND" in res["status"] or "ESCALATE" in res["status"]:
            try:
               # ==========================================
# 4. EXECUTE ALL SCENARIOS & GENERATE REPORT
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("Starting Part D Master Pipeline Execution")
    print("==========================================")
    
    if os.path.exists(RUNS_JSONL_PATH):
        os.remove(RUNS_JSONL_PATH)
        
    scenarios_df = pd.read_csv(SCENARIOS_FILE)
    summary_rows = []
    
    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        
        res = run_agent_master(msg, sid, run_id=f"run_{sid.lower()}")
        
        # Extract refund amount from final answer if present
        refund_amt = "N/A"
        if "REFUND" in res["status"] or "ESCALATE" in res["status"]:
            try:
                parts = res["final_answer"].split()
                for p in parts:
                    try:
                        refund_amt = f"INR {float(p)}"
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
                
        summary_rows.append({
            "Scenario ID": sid,
            "Status": res["status"],
            "Refund Amount": refund_amt,
            "Model Calls": res["llm_calls"],
            "Tokens Used": res["tokens_used"]
        })
        
        with open(RUNS_JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
            
        print(f"Completed {sid} -> Status: {res['status']} | Calls: {res['llm_calls']} | Tokens: {res['tokens_used']}")
        
    print("\n==========================================")
    print("Part D Summary Table (for README):")
    print("==========================================")
    
    # Custom markdown table printer (No 'tabulate' dependency required)
    headers = ["Scenario ID", "Status", "Refund Amount", "Model Calls", "Tokens Used"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|----" * len(headers) + "|")
    for row in summary_rows:
        print(f"| {row['Scenario ID']} | {row['Status']} | {row['Refund Amount']} | {row['Model Calls']} | {row['Tokens Used']} |")
