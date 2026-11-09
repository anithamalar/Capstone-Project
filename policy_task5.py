import os
import glob
import json
import sqlite3
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICIES_DIR = "data/policies"
DB_PATH = "data/instamart.db"
OUTPUTS_DIR = "outputs"
LEDGER_PATH = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- 1. Setup Retrieval Corpus & Vectorizer ---
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
    """Search policy documents and return top relevant chunks."""
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

# --- 2. Database Order Lookup ---
def get_order(order_id):
    """Retrieve order details from the instamart SQLite database."""
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

# --- 3. Refund Calculation ---
def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund amount based on delivery time and completion status."""
    # Refund rules: completed order delivered in > 45 minutes gets 100%, else 0.
    if delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0.0
    return {"refund_inr": refund}

# --- 4. Issue Refund / Escalate ---
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
    """Issue approved refund and log to ledger."""
    append_to_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    """Escalate refund request above INR 1000 to human review and log to ledger."""
    append_to_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

# --- 5. TOOLS_SPEC and TOOL_MAP ---
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

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

if __name__ == "__main__":
    print("--- Testing Tool Map ---")
    print("Registered Tools:", list(TOOL_MAP.keys()))
    
    # Test get_order if DB exists
    test_ord = get_order("ORD-000001")
    print(f"Test get_order('ORD-000001'): {test_ord}")
    
    # Test calculate_refund
    test_ref = calculate_refund(1200.0, 50)
    print(f"Test calculate_refund(1200.0, 50): {test_ref}")
