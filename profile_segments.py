import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. Load RFM table
rfm_path = "data/clean/rfm_table.csv"
print(f"Loading RFM data from {rfm_path}...")
df = pd.read_csv(rfm_path)

# 2. Standardize and fit K-Means with K=4
features = ["recency", "frequency", "monetary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
df["segment_id"] = kmeans.fit_predict(X_scaled)

# 3. Profile Segments (Size, Mean Recency, Frequency, Monetary)
print("\n--- (a) Segment Profiles ---")
profile = df.groupby("segment_id")[features].agg(["count", "mean"]).round(2)
print(profile)

# 4. Assign Business Names and Identify Premium Segment
# Note: Cluster numbers are arbitrary labels with no inherent ordering.
# Based on reference metrics: 
# - Highest monetary (~7,130) is the Premium/Champions segment.
# - High frequency (~19.4) & low recency (~8.8 days) is the Loyal/Active segment.
highest_monetary_cluster = df.groupby("segment_id")["monetary"].mean().idxmax()

segment_names = {}
is_premium_map = {}

for cid in df["segment_id"].unique():
    if cid == highest_monetary_cluster:
        segment_names[cid] = "Champions & VIPs"
        is_premium_map[cid] = True
    else:
        # Assign names based on RFM behavior patterns
        mean_freq = df[df["segment_id"] == cid]["frequency"].mean()
        mean_rec = df[df["segment_id"] == cid]["recency"].mean()
        if mean_freq > 12:
            segment_names[cid] = "Loyal Regulars"
            is_premium_map[cid] = False
        elif mean_rec > 50:
            segment_names[cid] = "At-Risk / Lapsed"
            is_premium_map[cid] = False
        else:
            segment_names[cid] = "Recent Starters"
            is_premium_map[cid] = False

df["segment_name"] = df["segment_id"].map(segment_names)
df["is_premium"] = df["segment_id"].map(is_premium_map)

print("\n--- (b) Segment Business Names & Premium Status Assigned ---")
for cid in sorted(df["segment_id"].unique()):
    name = segment_names[cid]
    prem = is_premium_map[cid]
    size = (df["segment_id"] == cid).sum()
    print(f"Cluster {cid} -> Name: '{name}' | Size: {size} customers | Premium: {prem}")

# 5. Apply PCA (n_components=2)
pca = PCA(n_components=2)
pca.fit(X_scaled)
evr = pca.explained_variance_ratio_

print("\n--- (c) PCA Explained Variance Ratio ---")
print(f"PC1: {evr[0]:.3f} | PC2: {evr[1]:.3f} | Total Explained Variance: {evr.sum():.3f}")

# 6. Export Final Dataset
output_dir = "data/clean"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "customer_segments.csv")
df.to_csv(output_path, index=False)
print(f"\n-> Final customer segments successfully saved to {output_path} ({len(df)} records)")
