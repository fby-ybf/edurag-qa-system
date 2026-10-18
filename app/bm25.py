"""BM25Okapi 精简实现（约 40 行，无第三方依赖）。"""
from __future__ import annotations

import math
from collections import Counter


class BM25Okapi:
    """标准 BM25：k1=1.5, b=0.75。

    中文按字符二元切分，避免依赖分词器。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.corpus: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.df: Counter[str] = Counter()
        self.n_docs = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = "".join(ch for ch in text if not ch.isspace() and ch.isalnum())
        return [text[i : i + 2] for i in range(max(0, len(text) - 1))] or [text]

    def fit(self, corpus: list[str]) -> "BM25Okapi":
        self.corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.n_docs = len(self.corpus)
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1
        return self

    def _idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))

    def score(self, query: str) -> list[float]:
        q_terms = self._tokenize(query)
        scores: list[float] = []
        for doc, dl in zip(self.corpus, self.doc_len):
            tf = Counter(doc)
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += self._idf(term) * f * (self.k1 + 1) / denom
            scores.append(s)
        return scores
