from typing import Dict, List, Tuple

def hybrid_rank(
    bm25_scores: Dict[str, float],
    dense_scores: Dict[str, float],
    alpha: float = 0.6
) -> List[Tuple[str, float]]:
    combined = {}
    for doc_id, b in bm25_scores.items():
        d = dense_scores.get(doc_id, 0.0)
        combined[doc_id] = alpha * b + (1 - alpha) * d

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)