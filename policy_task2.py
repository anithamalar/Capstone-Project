import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# 1. Load and Chunk Documents
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

# 2. Fit TF-IDF Vectorizer on all chunks
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)

# 3. Implement retrieve(query, k=3)
def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    # Get top-k indices sorted by similarity score descending
    top_indices = np.argsort(similarities)[::-1][:k]
    
    results = []
    for idx in top_indices:
        results.append({
            "source": all_chunks[idx]["source"],
            "score": float(similarities[idx]),
            "text": all_chunks[idx]["text"]
        })
    return results

# 4. Run Reference Test Query
if __name__ == "__main__":
    test_query = "refund policy late delivery"
    print(f"--- Running Retrieval for Query: '{test_query}' ---")
    top_chunks = retrieve(test_query, k=3)
    
    for rank, res in enumerate(top_chunks, 1):
        print(f"\nRank {rank}: [Score: {res['score']:.4f}] ({res['source']})")
        print(f"Text: {res['text']}")
