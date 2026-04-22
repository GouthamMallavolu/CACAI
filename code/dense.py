from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def _corpus_fingerprint(doc_ids: List[str]) -> str:
    h = hashlib.md5()
    for d in doc_ids[:2000]:
        h.update(str(d).encode("utf-8"))
    h.update(str(len(doc_ids)).encode("utf-8"))
    return h.hexdigest()

class DenseRetriever:
    def __init__(self, corpus: Dict[str, str], cache_dir: str | Path = "../outputs"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_ids = [str(x) for x in corpus.keys()]
        self.texts = list(corpus.values())

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.emb_path = cache_dir / f"embeddings.npy"

        if self.emb_path.exists():
            self.embeddings = np.load(self.emb_path)
        else:
            self.embeddings = self.model.encode(
                self.texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=64,
            )
            np.save(self.emb_path, self.embeddings)

        faiss.normalize_L2(self.embeddings)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            results.append((self.doc_ids[idx], float(score)))
        return results