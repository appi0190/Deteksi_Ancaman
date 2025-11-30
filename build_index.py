import pandas as pd
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATASET_PATH = "data/cyber_threat_intel_dataset.csv"

df = pd.read_csv(DATASET_PATH)

def build_document(row):
    return (
        f"Attack Type: {row['attack_type']}. "
        f"Platform: {row['platform']}. "
        f"Description: {row['description']}. "
        f"Impact: {row['impact']}. "
        f"Mitigation: {row['mitigation']}. "
        f"Source: {row['source']}."
    )

documents = [build_document(r) for _, r in df.iterrows()]
tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)
pickle.dump(bm25, open("indices/bm25_index.pkl", "wb"))
pickle.dump(documents, open("indices/documents.pkl", "wb"))

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True)
np.save("indices/embeddings.npy", embeddings)

print("BM25 & BERT indexes built successfully.")
