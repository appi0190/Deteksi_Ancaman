# evaluate.py (complete file)
import pandas as pd
import numpy as np
import os
import sys

K = 5  # ubah kalau mau Precision@3 atau @10

# -------------------------
# Utility metric functions
# -------------------------
def precision_at_k(relevances, k=K):
    r = list(relevances)[:k]
    if len(r) < k:
        r = r + [0] * (k - len(r))
    return float(sum(r)) / k

def recall_at_k(relevances, total_relevant, k=K):
    if total_relevant == 0:
        return 0.0
    r = list(relevances)[:k]
    return float(sum(r)) / total_relevant

def average_precision(relevances):
    num_rel = 0
    ap_sum = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel == 1:
            num_rel += 1
            ap_sum += num_rel / i
    if num_rel == 0:
        return 0.0
    return ap_sum / num_rel

# -------------------------
# Clean & load annotation
# -------------------------
def load_and_clean_annotation(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")

    df = pd.read_excel(path)

    # Normalize column names to lower for safety
    df.columns = [c.strip() for c in df.columns]

    # Try to find relevancy column among common variants
    possible_relev_cols = ['is_relevant', 'relevant', 'relevan(1/0)', 'relevan', 'relevance', 'label']
    found_relev = None
    for col in df.columns:
        if col.lower() in [x.lower() for x in possible_relev_cols]:
            found_relev = col
            break

    if found_relev is None:
        raise ValueError(f"Could not find relevancy column in {path}. Expected one of: {possible_relev_cols}. Columns present: {list(df.columns)}")

    # Coerce relevancy to numeric 0/1
    df[found_relev] = pd.to_numeric(df[found_relev], errors='coerce')

    # Fill NA with 0 (not relevant) — safer than dropping
    na_count = int(df[found_relev].isna().sum())
    if na_count > 0:
        print(f"[INFO] {path}: {na_count} NaN found in '{found_relev}' -> filling with 0 (not relevant).")

    df[found_relev] = df[found_relev].fillna(0).astype(int)

    # Ensure 'query' and 'rank' exist
    if 'query' not in df.columns:
        # try alternatives
        alternatives = ['query_text', 'q', 'query_id']
        for a in alternatives:
            if a in df.columns:
                df = df.rename(columns={a: 'query'})
                print(f"[INFO] Renamed column {a} -> 'query'")
                break
    if 'query' not in df.columns:
        raise ValueError(f"Annotation file {path} must contain a 'query' column (or one of alternatives). Columns: {list(df.columns)}")

    if 'rank' not in df.columns:
        # try to infer rank if there is a 'doc_rank' or we can compute rank by group order
        alternatives = ['doc_rank', 'ranked', 'position']
        for a in alternatives:
            if a in df.columns:
                df = df.rename(columns={a: 'rank'})
                print(f"[INFO] Renamed column {a} -> 'rank'")
                break

    # If still no 'rank', try to create rank by group order if there is no explicit rank
    if 'rank' not in df.columns:
        print(f"[INFO] No 'rank' column found in {path}. Creating 'rank' by order within each query group (assumes file already ordered).")
        df['rank'] = df.groupby('query').cumcount() + 1

    # Finally rename relev column to 'is_relevant' for uniformity
    df = df.rename(columns={found_relev: 'is_relevant'})

    # sort by query & rank
    df = df.sort_values(['query', 'rank']).reset_index(drop=True)

    return df

# -------------------------
# Compute metrics
# -------------------------
def compute_metrics(df, k=K):
    rows = []
    precision_list = []
    recall_list = []
    ap_list = []

    grouped = df.groupby('query')
    for query, g in grouped:
        g = g.sort_values('rank')
        relevances = g['is_relevant'].astype(int).tolist()
        total_relevant = sum(relevances)

        p_at_k = precision_at_k(relevances, k)
        r_at_k = recall_at_k(relevances, total_relevant, k)
        ap = average_precision(relevances)

        rows.append({
            'query': query,
            f'P@{k}': p_at_k,
            f'R@{k}': r_at_k,
            'AP': ap,
            'total_relevant': int(total_relevant),
            'num_returned': len(relevances)
        })

        precision_list.append(p_at_k)
        recall_list.append(r_at_k)
        ap_list.append(ap)

    # If lists are empty, avoid mean of empty -> set NaN explicitly
    mean_precision = float(np.mean(precision_list)) if len(precision_list) > 0 else float('nan')
    mean_recall = float(np.mean(recall_list)) if len(recall_list) > 0 else float('nan')
    map_score = float(np.mean(ap_list)) if len(ap_list) > 0 else float('nan')

    df_res = pd.DataFrame(rows)
    return df_res, {'mean_precision': mean_precision, 'mean_recall': mean_recall, 'MAP': map_score}

# -------------------------
# Main runner
# -------------------------
if __name__ == "__main__":
    bm25_file = "bm25_annotation.xlsx"
    bert_file = "bert_annotation.xlsx"

    # BM25
    try:
        bm25_df = load_and_clean_annotation(bm25_file)
        bm25_per_query, bm25_overall = compute_metrics(bm25_df, k=K)
        print("\n=== BM25 per-query (first 10 rows) ===")
        print(bm25_per_query.head(15).to_string(index=False))
        print("\n=== BM25 OVERALL ===")
        print(f"Precision@{K} : {bm25_overall['mean_precision']:.6f}")
        print(f"Recall@{K}    : {bm25_overall['mean_recall']:.6f}")
        print(f"MAP         : {bm25_overall['MAP']:.6f}")
    except Exception as e:
        print(f"[ERROR] BM25 evaluation failed: {e}")

    # BERT
    try:
        bert_df = load_and_clean_annotation(bert_file)
        bert_per_query, bert_overall = compute_metrics(bert_df, k=K)
        print("\n=== BERT per-query (first 10 rows) ===")
        print(bert_per_query.head(15).to_string(index=False))
        print("\n=== BERT OVERALL ===")
        print(f"Precision@{K} : {bert_overall['mean_precision']:.6f}")
        print(f"Recall@{K}    : {bert_overall['mean_recall']:.6f}")
        print(f"MAP         : {bert_overall['MAP']:.6f}")
    except Exception as e:
        print(f"[ERROR] BERT evaluation failed: {e}")
        sys.exit(1)