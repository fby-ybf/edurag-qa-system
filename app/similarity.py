"""中文相似度：字符 n-gram TF-IDF 余弦相似度。"""
from __future__ import annotations

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize(text: str) -> str:
    """去空白与标点，统一小写，降低分词差异影响。"""
    return re.sub(r"[\s\W]+", "", text.lower())


class Similarity:
    """对一批候选文本计算与查询的余弦相似度。"""

    def __init__(self, ngram_range: tuple[int, int] = (2, 3)) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        self._matrix = None

    def fit(self, texts: list[str]) -> "Similarity":
        self._matrix = self.vectorizer.fit_transform([normalize(t) for t in texts])
        return self

    def score(self, query: str) -> list[float]:
        if self._matrix is None:
            raise RuntimeError("请先调用 fit()")
        qv = self.vectorizer.transform([normalize(query)])
        return cosine_similarity(qv, self._matrix)[0].tolist()
