from flask import Flask, render_template, request
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np

# -------------------------
# CONFIG (tweak threshold di sini)
# -------------------------
TORCH_DEVICE = "cpu"
BERT_THRESHOLD = 0.45     # jika max cosine < ini => anggap ngawur (naikkan kalau masih false positive)
HYBRID_THRESHOLD = 0.12   # jika hybrid terbaik < ini => anggap ngawur (safety)
BM25_MIN_SCORE_FOR_ACCEPT = 3.0  # kalau BM25 best raw score > ini, boleh terima meskipun BERT lemah (opsional)

# -------------------------
# Inisialisasi model & data
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE)

DATA_PATH = "data/cyber_threat_intel_datasett.csv"
df = pd.read_csv(DATA_PATH)

TEXT_PRIORITY = [
    "title", "description", "tags", "threat_type", "severity",
    "attack_type", "platform", "impact", "mitigation", "source",
]

def combine_columns(row):
    vals = []
    for c in TEXT_PRIORITY:
        if c in row and pd.notna(row[c]):
            t = str(row[c]).strip()
            if t:
                vals.append(t)
    return " ".join(vals)

df["combined_text"] = df.apply(combine_columns, axis=1)
documents = df["combined_text"].astype(str).tolist()

# build BM25
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# build BERT embeddings (once)
print("🔄 Building/Loading BERT embeddings (CPU)...")
emb_np = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
emb_tensor = torch.tensor(emb_np, device=TORCH_DEVICE)
print("✅ Embeddings ready.")

# display helpers
DISPLAY_PRIORITY = {
    "title": ["title", "attack_type", "id"],
    "description": ["description", "impact", "mitigation"],
    "tags": ["tags", "platform", "source"],
    "threat_type": ["threat_type", "attack_type", "platform"],
    "severity": ["severity", "impact", "mitigation"],
}

def get_first_available(row, candidates):
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            s = str(row[col]).strip()
            if s:
                return s
    return ""

def build_result(idx):
    row = df.loc[idx]
    return {
        "title": get_first_available(row, DISPLAY_PRIORITY["title"]),
        "description": get_first_available(row, DISPLAY_PRIORITY["description"]),
        "tags": get_first_available(row, DISPLAY_PRIORITY["tags"]),
        "threat_type": get_first_available(row, DISPLAY_PRIORITY["threat_type"]),
        "severity": get_first_available(row, DISPLAY_PRIORITY["severity"]),
        "full_text": documents[idx],
    }

# normalizer helper (avoid div by zero)
def normalize_array(arr):
    a = np.array(arr, dtype=float)
    amin, amax = a.min(), a.max()
    if amax - amin < 1e-9:
        return np.zeros_like(a)  # all equal -> return zeros
    return (a - amin) / (amax - amin)

# -------------------------
# SEARCH FUNCTION (improved)
# -------------------------
def search(query):
    q = (query or "").strip()
    if q == "":
        return None

    # BM25 raw scores (for all docs)
    tokens = q.split()
    bm25_scores = bm25.get_scores(tokens)  # numpy array

    # BERT scores (cosine similarity)
    q_emb = model.encode(q, convert_to_tensor=True, device=TORCH_DEVICE)
    bert_scores_tensor = util.cos_sim(q_emb, emb_tensor)[0]
    bert_scores = bert_scores_tensor.cpu().numpy()  # numpy array

    # quick diagnostics
    max_bert = float(np.max(bert_scores))
    max_bm25 = float(np.max(bm25_scores))

    # If semantic match is very low -> NO RESULT
    if max_bert < BERT_THRESHOLD and max_bm25 < BM25_MIN_SCORE_FOR_ACCEPT:
        # log for debugging/tuning
        print(f"[NO RESULT] query='{q}' max_bert={max_bert:.4f} max_bm25={max_bm25:.4f}")
        return "none"

    # Hybrid scoring: normalize only on candidate pool to be efficient
    # Pool: top-N BM25 + top-N BERT
    TOP_N = 50
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:TOP_N]
    bert_top_idx = np.argsort(bert_scores)[::-1][:TOP_N]
    pool = list(np.unique(np.concatenate([bm25_top_idx, bert_top_idx])))

    pool_bm25 = bm25_scores[pool]
    pool_bert = bert_scores[pool]

    norm_bm25 = normalize_array(pool_bm25)
    norm_bert = normalize_array(pool_bert)

    # compute hybrid score for pool
    hybrid_scores = (0.35 * norm_bm25) + (0.65 * norm_bert)  # heavier to BERT

    # pick best in pool
    best_idx_in_pool = np.argmax(hybrid_scores)
    best_doc_idx = int(pool[best_idx_in_pool])
    best_hybrid_score = float(hybrid_scores[best_idx_in_pool])
    best_doc_bert = float(pool_bert[best_idx_in_pool])
    best_doc_bm25 = float(pool_bm25[best_idx_in_pool])

    # final safety threshold on hybrid
    if best_hybrid_score < HYBRID_THRESHOLD and max_bm25 < BM25_MIN_SCORE_FOR_ACCEPT:
        print(f"[NO RESULT-HYBRID] query='{q}' hybrid_best={best_hybrid_score:.4f} max_bert={max_bert:.4f} max_bm25={max_bm25:.4f}")
        return "none"

    # else build result
    result = build_result(best_doc_idx)
    # attach debug scores (optional)
    result["_scores"] = {
        "hybrid": round(best_hybrid_score, 4),
        "bert": round(best_doc_bert, 4),
        "bm25": round(best_doc_bm25, 4),
        "max_bert": round(max_bert, 4),
        "max_bm25": round(max_bm25, 4)
    }

    print(f"[HIT] query='{q}' idx={best_doc_idx} hybrid={best_hybrid_score:.4f} bert={best_doc_bert:.4f} bm25={best_doc_bm25:.4f}")

    return result

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    result = None
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            result = search(query)

    return render_template("index.html", result=result, query=query)

if __name__ == "__main__":
    app.run(debug=True)