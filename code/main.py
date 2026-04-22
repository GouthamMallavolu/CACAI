from __future__ import annotations
from pathlib import Path
import pandas as pd

from bm25 import BM25
from dense import DenseRetriever
from hybrid import hybrid_rank
from evaluation import precision_at_k, average_precision, ndcg_at_k

DATA_DIR = Path("../data")
OUT_DIR = Path("../outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_qrels_dict(qrels_df: pd.DataFrame):
    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)
    qrels_df["relevance"] = qrels_df["relevance"].astype(int)

    qrels = {}
    for qid, grp in qrels_df.groupby("query_id"):
        qrels[qid] = dict(zip(grp["doc_id"], grp["relevance"]))
    return qrels

def main():
    corpus_df = pd.read_csv(DATA_DIR / "corpus.csv")
    corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)
    corpus = dict(zip(corpus_df["doc_id"], corpus_df["text"]))

    queries_df = pd.read_csv(DATA_DIR / "queries.csv")
    queries_df["query_id"] = queries_df["query_id"].astype(str)

    qrels_df = pd.read_csv(DATA_DIR / "qrels_filtered.csv")
    qrels_dict = build_qrels_dict(qrels_df)

    bm25 = BM25(corpus)
    dense = DenseRetriever(corpus, cache_dir=OUT_DIR)

    rows = []
    # Evaluate first N queries
    eval_q = [qid for qid in queries_df["query_id"].tolist() if qid in qrels_dict][:50]

    for qid in eval_q:
        query = queries_df.loc[queries_df["query_id"] == qid, "query"].iloc[0]
        rel_dict = qrels_dict[qid]

        # BM25
        bm25_scores = {d: bm25.score(query, d) for d in corpus}
        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_docs = [d for d, _ in bm25_ranked]

        # Dense
        dense_ranked = dense.search(query, top_k=len(corpus))
        dense_docs = [d for d, _ in dense_ranked]

        # Hybrid
        dense_scores = dict(dense.search(query, top_k=1000))
        hybrid_ranked = hybrid_rank(bm25_scores, dense_scores, alpha=0.6)
        hybrid_docs = [d for d, _ in hybrid_ranked]

        for method, docs in [("BM25", bm25_docs), ("Dense", dense_docs), ("Hybrid", hybrid_docs)]:
            rows.append({
                "query_id": qid,
                "method": method,
                "P@10": precision_at_k(docs, rel_dict, k=10),
                "MAP": average_precision(docs, rel_dict),
                "nDCG@10": ndcg_at_k(docs, rel_dict, k=10)
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    print(out.groupby("method").mean(numeric_only=True))

if __name__ == "__main__":
    main()