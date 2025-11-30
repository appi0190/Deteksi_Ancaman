import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from queries import queries

# Load dataset
df = pd.read_csv("data/cyber_threat_intel_datasett.csv")
documents = df["description"].astype(str).tolist()

# Load BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode all documents
doc_embeddings = model.encode(documents, convert_to_tensor=True)

def bert_search(query, k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, doc_embeddings)[0]
    top_k_idx = torch.topk(scores, k).indices
    return top_k_idx.cpu().numpy(), scores[top_k_idx].cpu().numpy()


# === Evaluate all queries ===
def evaluate_bert(k=5):
    results = []

    all_queries = (
        [("literal", q) for q in queries["literal"]] +
        [("descriptive", q) for q in queries["descriptive"]] +
        [("ngawur", q) for q in queries["ngawur"]]
    )

    for category, q in all_queries:
        idx, scores = bert_search(q, k=k)
        results.append([category, q, idx])

    return pd.DataFrame(results, columns=["category", "query", "top_k_docs"])


if __name__ == "__main__":
    df_result = evaluate_bert()
    print(df_result)
