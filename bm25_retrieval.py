import pandas as pd
from rank_bm25 import BM25Okapi
from queries import queries

# Load dataset
df = pd.read_csv("data/cyber_threat_intel_datasett.csv")
documents = df["description"].astype(str).tolist()

# Preprocessing simple (lowercase)
tokenized_docs = [doc.lower().split() for doc in documents]

# Build BM25 model
bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query, k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_k_idx = scores.argsort()[-k:][::-1]
    return top_k_idx, [scores[i] for i in top_k_idx]


# === Evaluate all queries ===
def evaluate_bm25(k=5):
    results = []

    # Gabungkan semua query
    all_queries = (
        [("literal", q) for q in queries["literal"]] +
        [("descriptive", q) for q in queries["descriptive"]] +
        [("ngawur", q) for q in queries["ngawur"]]
    )

    for category, q in all_queries:
        idx, scores = bm25_search(q, k=k)
        results.append([category, q, idx])

    return pd.DataFrame(results, columns=["category", "query", "top_k_docs"])


if __name__ == "__main__":
    df_result = evaluate_bm25()
    print(df_result)
