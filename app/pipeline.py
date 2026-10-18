"""问答编排：FAQ 阈值直答，未命中走双路召回；统计与反馈沉淀。"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from .faq_store import FaqStore
from .knowledge import KnowledgeBase, load_chunks


class QaPipeline:
    def __init__(self, faq_path: str | Path, kb_path: str | Path) -> None:
        self.faq = FaqStore(faq_path)
        self.kb = KnowledgeBase(load_chunks(kb_path))
        self.feedback: list[dict] = []
        self.unresolved: Counter[str] = Counter()
        self.total = 0
        self.faq_hits = 0
        self.total_ms = 0.0

    def ask(self, question: str, top_k: int = 3) -> dict:
        start = time.perf_counter()
        self.total += 1
        hit = self.faq.match(question)
        if hit:
            self.faq_hits += 1
            answer, source, score = hit["answer"], "faq", hit["score"]
            contexts = []
        else:
            contexts = self.kb.search(question, top_k=top_k)
            self.unresolved[question] += 1
            if contexts:
                answer = "；".join(c["text"][:90] for c in contexts[:2])
                source, score = "knowledge", contexts[0]["score"]
            else:
                answer, source, score = "知识库中暂无相关内容，已记录待补充。", "none", 0.0
        self.total_ms += (time.perf_counter() - start) * 1000
        return {
            "question": question,
            "answer": answer,
            "source": source,
            "score": score,
            "contexts": contexts,
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    def add_feedback(self, question: str, useful: bool) -> dict:
        self.feedback.append({"question": question, "useful": useful})
        return {"recorded": True, "total_feedback": len(self.feedback)}

    def stats(self) -> dict:
        return {
            "total_questions": self.total,
            "faq_hits": self.faq_hits,
            "faq_hit_rate": round(self.faq_hits / self.total, 4) if self.total else 0.0,
            "avg_ms": round(self.total_ms / self.total, 2) if self.total else 0.0,
            "top_unresolved": self.unresolved.most_common(5),
        }
