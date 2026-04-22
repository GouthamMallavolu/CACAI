import math
from collections import Counter, defaultdict
from typing import Dict

class BM25:
    def __init__(self, corpus: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.doc_len = {}
        self.avgdl = 0.0
        self.df = defaultdict(int)
        self.idf = {}
        self._init_stats()

    def _init_stats(self):
        total_len = 0
        for doc_id, text in self.corpus.items():
            terms = text.split()
            self.doc_len[doc_id] = len(terms)
            total_len += len(terms)
            for term in set(terms):
                self.df[term] += 1

        self.avgdl = total_len / max(self.N, 1)

        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        score = 0.0
        q_terms = query.split()
        d_terms = self.corpus[doc_id].split()
        tf = Counter(d_terms)

        dl = self.doc_len[doc_id]
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl))

        for t in q_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + denom_norm
            score += self.idf.get(t, 0.0) * (numerator / denominator)

        return score