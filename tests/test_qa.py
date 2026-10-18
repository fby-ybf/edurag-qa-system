"""FAQ 阈值直答 / 双路召回 / 统计 / API 测试。"""
from fastapi.testclient import TestClient

from app.api import app
from app.bm25 import BM25Okapi
from app.faq_store import FaqStore
from app.knowledge import load_chunks
from app.pipeline import QaPipeline
from app.similarity import Similarity

FAQ = [
    {"question": "怎么选课？", "answer": "教务系统选课。"},
    {"question": "四六级报名时间", "answer": "3 月和 9 月报名。"},
]


def test_bm25_ranks_keyword_doc_first():
    bm25 = BM25Okapi().fit(["今天天气很好适合出门", "数据库索引原理与实践"])
    scores = bm25.score("数据库索引")
    assert scores[1] > scores[0]


def test_similarity_threshold_hit():
    sim = Similarity().fit(["怎么选课？", "四六级报名时间"])
    scores = sim.score("如何选课")
    assert scores[0] >= 0.5


def test_faq_threshold_hit_returns_standard_answer(tmp_path):
    import json

    p = tmp_path / "faq.json"
    p.write_text(json.dumps(FAQ, ensure_ascii=False), encoding="utf-8")
    kb = tmp_path / "kb.txt"
    kb.write_text("选课在教务系统进行，分为预选与正选两个阶段。", encoding="utf-8")
    pipe = QaPipeline(p, kb)
    result = pipe.ask("怎么选课")
    assert result["source"] == "faq"
    assert result["answer"] == "教务系统选课。"


def test_unmatched_question_falls_back_to_knowledge(tmp_path):
    import json

    p = tmp_path / "faq.json"
    p.write_text(json.dumps(FAQ, ensure_ascii=False), encoding="utf-8")
    kb = tmp_path / "kb.txt"
    kb.write_text(
        "学籍管理：本科生标准学制四年，最长学习年限六年，提前毕业需学院审核。",
        encoding="utf-8",
    )
    pipe = QaPipeline(p, kb)
    result = pipe.ask("最长学习年限是几年")
    assert result["source"] == "knowledge"
    assert result["contexts"], "应召回知识库片段"
    assert "六年" in result["answer"]


def test_stats_tracks_hit_rate(tmp_path):
    import json

    p = tmp_path / "faq.json"
    p.write_text(json.dumps(FAQ, ensure_ascii=False), encoding="utf-8")
    kb = tmp_path / "kb.txt"
    kb.write_text("校园卡丢失后应先挂失再补办，补办工本费 20 元。", encoding="utf-8")
    pipe = QaPipeline(p, kb)
    pipe.ask("怎么选课")
    pipe.ask("校园卡丢了怎么办")
    stats = pipe.stats()
    assert stats["total_questions"] == 2
    assert stats["faq_hits"] == 1
    assert stats["faq_hit_rate"] == 0.5


def test_api_end_to_end():
    client = TestClient(app)
    assert client.get("/healthz").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "怎么选课？"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "faq"
    assert body["answer"]
    # 该问题不在 FAQ 库中，应走知识库双路召回
    r = client.post("/ask", json={"question": "创新创业学分怎么认定"})
    assert r.json()["source"] in ("knowledge", "none")
    assert client.post("/feedback", json={"question": "怎么选课？", "useful": True}).json()["recorded"]
    assert client.get("/stats").json()["total_questions"] >= 2
