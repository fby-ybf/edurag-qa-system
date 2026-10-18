"""知识库：分块 + 向量 / BM25 双路召回融合排序。"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .bm25 import BM25Okapi
from .similarity import Similarity

CHUNK_SIZE, CHUNK_OVERLAP = 260, 40


def load_chunks(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for para in paras:
        if len(para) <= CHUNK_SIZE:
            chunks.append(para)
            continue
        step = CHUNK_SIZE - CHUNK_OVERLAP
        chunks.extend(para[i : i + CHUNK_SIZE] for i in range(0, len(para), step))
    return chunks


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class KnowledgeBase:
    """双路召回：TF-IDF 向量（语义相近）+ BM25（关键词精确），归一化后加权融合。"""

    def __init__(self, chunks: list[str], vector_weight: float = 0.6) -> None:
        self.chunks = chunks
        self.vector_weight = vector_weight
        self.sim = Similarity().fit(chunks)
        self.bm25 = BM25Okapi().fit(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        vec = minmax(self.sim.score(query))
        kw = minmax(self.bm25.score(query))
        fused = [
            self.vector_weight * vec[i] + (1 - self.vector_weight) * kw[i] for i in range(len(self.chunks))
        ]
        order = np.argsort(-np.array(fused))[:top_k]
        return [
            {
                "text": self.chunks[int(i)],
                "score": round(float(fused[int(i)]), 4),
                "vector_score": round(float(vec[int(i)]), 4),
                "bm25_score": round(float(kw[int(i)]), 4),
            }
            for i in order
        ]
