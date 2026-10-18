"""FAQ 库：加载、相似度匹配、阈值直答。"""
from __future__ import annotations

import json
from pathlib import Path

from .similarity import Similarity

DEFAULT_THRESHOLD = 0.85


class FaqStore:
    def __init__(self, path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.path = Path(path)
        self.threshold = threshold
        self.items: list[dict] = json.loads(self.path.read_text(encoding="utf-8"))
        self.questions = [item["question"] for item in self.items]
        self.sim = Similarity().fit(self.questions)

    def match(self, query: str) -> dict | None:
        """相似度超过阈值返回 {'score', 'answer', 'question'}，否则 None。"""
        if not self.items:
            return None
        scores = self.sim.score(query)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] >= self.threshold:
            return {
                "score": round(float(scores[best_idx]), 4),
                "question": self.questions[best_idx],
                "answer": self.items[best_idx]["answer"],
            }
        return None

    def all(self) -> list[dict]:
        return self.items
