import os
import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICIES_DIR = "data/policies"
EVAL_FILE = "data/raw/policy_eval.csv"

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
            all_chunks.append({
                "source": filename,
                "text": chunk
            })
    return all_chunks

def evaluate_pipeline(chunk_size=40, overlap=10):
    chunks = build_corpus(chunk_size=chunk_size, overlap=overlap)
    chunk_texts = [c["text"] for c in chunks]
    
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunk_texts)
    
    def retrieve(query, k=3):
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, chunk_vectors).flatten()
        top_indices = np.argsort(sims)[::-1][:k]
        return [chunks[idx] for idx in top_indices]

    def answer(question):
        top_chunks = retrieve(question, k=3)
        context_parts = []
        sources = []
        for c in top_chunks:
            sources.append(c["source"])
            context_parts.append(f"[{c['source']}]\n{c['text']}")
            
        context_str = "\n".join(context_parts)
        user_content = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
        system_content = (
            "You are a POLICY ASSISTANT. Answer the user's question using ONLY "
            "the supplied context. If the answer cannot be found in the context, "
            "say that you could not find the information."
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, sources

    eval_df = pd.read_csv(EVAL_FILE)
    
    ans_retrieved_correct = 0
    ans_fact_correct = 0
    unans_fallback_correct = 0
    failed_questions = []

    for _, row in eval_df.iterrows():
        q = row["question"]
        expected_doc = str(row.get("expected_doc", "")).strip()
        expected_fact = str(row.get("expected_fact", "")).strip()
        is_answerable = bool(row.get("is_answerable", True))
        
        response_text, sources = answer(q)
        
        if is_answerable:
            # Check if expected_doc is in top 3 sources
            if expected_doc in sources:
                ans_retrieved_correct += 1
            else:
                failed_questions.append((q, "retrieval", sources, expected_doc))
                
            # Check if expected_fact is in response (case-sensitive substring)
            if expected_fact in response_text:
                ans_fact_correct += 1
            else:
                failed_questions.append((q, "fact", response_text, expected_fact))
        else:
            # Check unanswerable fallback (begins with "I could not find")
            if response_text.strip().lower().startswith("i could not find"):
                unans_fallback_correct += 1
            else:
                failed_questions.append((q, "fallback", response_text))

    return {
        "ans_retrieved": ans_retrieved_correct,
        "ans_fact": ans_fact_correct,
        "unans_fallback": unans_fallback_correct,
        "failures": failed_questions
    }

print("--- Evaluating with Overlap = 10 ---")
res_overlap10 = evaluate_pipeline(chunk_size=40, overlap=10)

print("--- Evaluating with Overlap = 0 ---")
res_overlap0 = evaluate_pipeline(chunk_size=40, overlap=0)

print("\n================ TASK 4 EVALUATION REPORT ================")
print(f"Overlap 10 (Reference Standard):")
print(f"  - Answerable Expected Doc Found in Top 3: {res_overlap10['ans_retrieved']} / 12")
print(f"  - Answerable Expected Fact Present: {res_overlap10['ans_fact']} / 12")
print(f"  - Unanswerable Fallback Triggered: {res_overlap10['unans_fallback']} / 2")

print(f"\nOverlap 0 (Zero Overlap):")
print(f"  - Answerable Expected Doc Found in Top 3: {res_overlap0['ans_retrieved']} / 12")
print(f"  - Answerable Expected Fact Present: {res_overlap0['ans_fact']} / 12")
print(f"  - Unanswerable Fallback Triggered: {res_overlap0['unans_fallback']} / 2")
if res_overlap0['failures']:
    print(f"\nSample failures with Overlap 0:")
    for f in res_overlap0['failures'][:2]:
        print(f"  - Question: {f[0]} | Type: {f[1]}")
