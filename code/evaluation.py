from __future__ import annotations
from typing import Dict, List
import numpy as np

def precision_at_k(ranked_docs: List[str], rel_dict: Dict[str, int], k: int = 10) -> float:
    topk = ranked_docs[:k]
    hits = sum(1 for d in topk if rel_dict.get(d, 0) > 0)
    return hits / k

def average_precision(ranked_docs: List[str], rel_dict: Dict[str, int]) -> float:
    num_rel_total = sum(1 for v in rel_dict.values() if v > 0)
    if num_rel_total == 0:
        return 0.0

    hits = 0
    s = 0.0
    for i, d in enumerate(ranked_docs, start=1):
        if rel_dict.get(d, 0) > 0:
            hits += 1
            s += hits / i
    return s / num_rel_total

def dcg_at_k(ranked_docs: List[str], rel_dict: Dict[str, int], k: int = 10) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked_docs[:k], start=1):
        rel = rel_dict.get(d, 0)
        if rel > 0:
            dcg += (2**rel - 1) / np.log2(i + 1)
    return dcg

def ndcg_at_k(ranked_docs: List[str], rel_dict: Dict[str, int], k: int = 10) -> float:
    dcg = dcg_at_k(ranked_docs, rel_dict, k)
    ideal_rels = sorted(rel_dict.values(), reverse=True)[:k]
    if not ideal_rels or sum(ideal_rels) == 0:
        return 0.0
    ideal_dcg = sum((2**rel - 1) / np.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0