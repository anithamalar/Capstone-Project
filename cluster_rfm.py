import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load RFM table from Part B Task 1
rfm_path = "data/clean/rfm_table.csv"
print(f"Loading RFM data from {rfm_path}...")
df = pd.read_csv(rfm_path)

# Select RFM columns for clustering
features = ["recency", "frequency", "monetary"]
X = df[features]

# Standardize features (mean = 0, variance = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- K-Means Evaluation (K = 2 to 8) ---")
results = []

for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, labels)
    
    results.append({
        "K": k,
        "Inertia": round(inertia, 1),
        "Silhouette Score": round(sil_score, 3)
    })

# Display the results table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select final K based on silhouette peak and business interpretability
# K=4 has the highest silhouette score (0.526) and offers 4 clear actionable segments (e.g., Champions, Loyal, At-Risk, Lapsed)
final_k = 4
print(f"\nSelected Optimal K: {final_k} (Optimal balance of silhouette score and business interpretability)")
