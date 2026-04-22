from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from bm25 import BM25
from dense import DenseRetriever
from hybrid import hybrid_rank
from rag import generate_answer
from reformulation import extract_topic_entity, reformulate_with_entity


class ConversationalIR:
    """
    Backend system used by Streamlit.
    - Keeps a topic_entity memory for conversational reformulation.
    - Supports BM25 / Dense / Hybrid.
    - Optional RAG answer generation from top retrieved passages.
    """

    def __init__(self, data_dir: str | Path = "../data", outputs_dir: str | Path = "../outputs"):
        data_dir = Path(data_dir)
        outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Load corpus
        corpus_df = pd.read_csv(data_dir / "corpus.csv")
        if "doc_id" not in corpus_df.columns or "text" not in corpus_df.columns:
            raise ValueError("corpus.csv must have columns: doc_id,text")

        corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)
        self.corpus: Dict[str, str] = dict(zip(corpus_df["doc_id"], corpus_df["text"]))

        # Init retrieval components
        self.bm25 = BM25(self.corpus)
        self.dense = DenseRetriever(self.corpus, cache_dir=outputs_dir)

        # Conversation memory
        self.topic_entity: Optional[str] = None

    def reset_conversation(self):
        self.topic_entity = None

    def search(
        self,
        query: str,
        method: str = "Hybrid",
        alpha: float = 0.6,
        use_reformulation: bool = True,
        use_rag: bool = False,
        top_k: int = 10,
        dense_topk_for_hybrid: int = 1000,
    ) -> tuple[str, List[Tuple[str, float]], Optional[str]]:
        """
        Returns:
          reformulated_query: str
          ranked: List[(doc_id, score)] top_k
          answer: Optional[str]
        """

        query = (query or "").strip()
        if not query:
            return "", [], None

        # --- Conversational Reformulation (topic entity memory) ---
        if use_reformulation:
            new_topic = extract_topic_entity(query)
            if new_topic:
                self.topic_entity = new_topic

            reformulated_query = reformulate_with_entity(query, self.topic_entity)
        else:
            reformulated_query = query

        # --- BM25 scores for ALL docs (needed for Hybrid fusion) ---
        bm25_scores = {doc_id: self.bm25.score(reformulated_query, doc_id) for doc_id in self.corpus}

        # --- Dense retrieval scores ---
        # For Hybrid, we grab more dense results to reduce sparsity.
        dense_results = self.dense.search(reformulated_query, top_k=max(top_k, dense_topk_for_hybrid))
        dense_scores = dict(dense_results)

        # --- Rank based on selected method ---
        m = method.strip().lower()
        if m == "bm25":
            ranked_all = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            ranked = ranked_all[:top_k]

        elif m == "dense":
            ranked = self.dense.search(reformulated_query, top_k=top_k)

        else:  # Hybrid default
            ranked = hybrid_rank(bm25_scores, dense_scores, alpha=alpha)[:top_k]

        # RAG
        answer = None
        if use_rag and ranked:
            top_contexts = [self.corpus.get(doc_id, "")[:450] for doc_id, _ in ranked[:3]]
            answer = generate_answer(reformulated_query, top_contexts, topic_entity=self.topic_entity)

        return reformulated_query, ranked, answer