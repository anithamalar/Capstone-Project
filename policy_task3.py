import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICIES_DIR = "data/policies"

def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if not words:
        return chunks
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks

# 1. Load, Chunk and Vectorize Documents (Reusing Task 1 & 2 logic)
all_chunks = []
policy_files = glob.glob(os.path.join(POLICIES_DIR, "*.txt"))

for file_path in sorted(policy_files):
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = chunk_text(content, chunk_size=40, overlap=10)
    for idx, chunk in enumerate(chunks):
        all_chunks.append({
            "source": filename,
            "chunk_index": idx,
            "text": chunk
        })

chunk_texts = [c["text"] for c in all_chunks]
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)

def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "source": all_chunks[idx]["source"],
            "score": float(similarities[idx]),
            "text": all_chunks[idx]["text"]
        })
    return results

# 2. Implement answer(question) for Task 3
def answer(question):
    # Retrieve top 3 chunks
    top_chunks = retrieve(question, k=3)
    
    # Construct context string with exact formatting requirements
    context_parts = []
    sources = set()
    for chunk in top_chunks:
        sources.add(chunk["source"])
        context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
    
    context_str = "\n".join(context_parts)
    
    # Construct user message
    user_content = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
    
    # System message containing the words POLICY ASSISTANT
    system_content = (
        "You are a POLICY ASSISTANT. Answer the user's question using ONLY "
        "the supplied context. If the answer cannot be found in the context, "
        "say that you could not find the information."
    )
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    # Call mock_llm
    response_obj = mock_llm.chat(messages)
    model_text = response_obj.choices[0].message.content
    
    return {
        "response": model_text,
        "sources": list(sources)
    }

# 3. Test the answer function
if __name__ == "__main__":
    test_q = "What is the policy for a late delivery refund?"
    print(f"--- Asking Question: '{test_q}' ---\n")
    
    result = answer(test_q)
    print("Model Response:")
    print(result["response"])
    print(f"\nSources Used: {result['sources']}")
