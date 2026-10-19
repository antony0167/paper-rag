"""HTTP API.

    POST /search  {"query": "...", "k": 5, "mode": "hybrid", "rerank": true}
    POST /ask     {"question": "...", "mode": "hybrid", "rerank": true}
    GET  /health
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm import LLMError


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=20)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    rerank: bool = True


def create_app(rag) -> FastAPI:
    app = FastAPI(title="paper-rag", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(rag.index.chunks)}

    @app.post("/search")
    def search(req: SearchRequest):
        hits = rag.retrieve(req.query, req.mode, req.rerank, k=req.k)
        return [{"rank": h.rank + 1, "score": round(h.score, 4), "chunk_id": h.chunk.chunk_id,
                 "arxiv_id": h.chunk.arxiv_id, "title": h.chunk.title, "page": h.chunk.page,
                 "text": h.chunk.text} for h in hits]

    @app.post("/ask")
    def ask(req: AskRequest):
        try:
            return rag.ask(req.question, req.mode, req.rerank).to_dict()
        except LLMError as e:
            raise HTTPException(status_code=502, detail=str(e))

    return app
