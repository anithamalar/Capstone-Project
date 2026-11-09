import os
import glob

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
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # Stop after the first chunk whose end reaches or passes the last word
        if end >= len(words):
            break
            
        start += step
        
    return chunks

print("--- Processing Policy Documents & Chunking ---")
all_chunks = []
chunk_counts = {}

# Find all .txt files in policies directory
policy_files = glob.glob(os.path.join(POLICIES_DIR, "*.txt"))

for file_path in sorted(policy_files):
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    chunks = chunk_text(content, chunk_size=40, overlap=10)
    chunk_counts[filename] = len(chunks)
    
    for idx, chunk in enumerate(chunks):
        all_chunks.append({
            "source": filename,
            "chunk_index": idx,
            "text": chunk
        })

print(f"\nTotal chunks created: {len(all_chunks)}")
print("Breakdown by file:")
for fname, count in chunk_counts.items():
    print(f"  - {fname}: {count} chunks")

print("\n--- Sample Chunks Preview ---")
for i, item in enumerate(all_chunks[:3]):
    print(f"\n[Chunk {i+1}] Source: {item['source']}")
    print(f"Text: {item['text'][:100]}...")
