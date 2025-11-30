import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from queries import queries  # pastikan file queries.py ada

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("data/cyber_threat_intel_datasett.csv")
documents = df["description"].astype(str).tolist()

# =========================
# 2. BM25 SETUP
# =========================
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

def bm25_search(query, k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_k_idx = scores.argsort()[-k:][::-1]
    return top_k_idx


# =========================
# 3. BERT SETUP
# =========================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents)

def bert_search(query, k=5):
    query_embed = model.encode([query])
    similarities = cosine_similarity(query_embed, doc_embeddings)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]
    return top_k_idx


# =========================
# 4. GENERATE EXCEL ANNOTATION FILES
# =========================
def generate_annotation_file(method_name, search_function, filename="output.xlsx", k=5):
    rows = []

    for category in queries:
        for query in queries[category]:

            # ambil top-k hasil
            top_idx = search_function(query, k=k)

            # masukkan ke tabel excel
            for rank, doc_idx in enumerate(top_idx, start=1):
                rows.append({
                    "category": category,
                    "query": query,
                    "rank": rank,
                    "doc_index": doc_idx,
                    "document_text": documents[doc_idx],
                    "relevan(1/0)": ""  # isi manual nanti
                })

    df_out = pd.DataFrame(rows)
    df_out.to_excel(filename, index=False)
    print(f"File berhasil dibuat: {filename}")


# =========================
# 5. RUN: BUAT DUA FILE ANOTASI
# =========================

generate_annotation_file(
    method_name="BM25",
    search_function=bm25_search,
    filename="bm25_annotation.xlsx"
)

generate_annotation_file(
    method_name="BERT",
    search_function=bert_search,
    filename="bert_annotation.xlsx"
)

print("Semua file anotasi sudah dibuat.")
