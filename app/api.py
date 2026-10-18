"""FastAPI 路由。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .pipeline import QaPipeline

BASE = Path(__file__).resolve().parents[1]
app = FastAPI(title="edurag-qa-system", version="1.0.0")
pipeline = QaPipeline(BASE / "data" / "faq.json", BASE / "data" / "knowledge.txt")


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class FeedbackRequest(BaseModel):
    question: str
    useful: bool


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "faq_size": len(pipeline.faq.items), "kb_chunks": len(pipeline.kb.chunks)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return pipeline.ask(req.question, top_k=req.top_k)


@app.get("/faqs")
def faqs() -> dict:
    return {"items": pipeline.faq.all(), "threshold": pipeline.faq.threshold}


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    return pipeline.add_feedback(req.question, req.useful)


@app.get("/stats")
def stats() -> dict:
    return pipeline.stats()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
