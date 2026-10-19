"""Retrieve -> (rerank) -> answer with citations."""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .index import Hit, Index

SYSTEM_PROMPT = """You answer questions about research papers using only the numbered sources you are given.
Rules:
- Cite every claim with the source number in square brackets, e.g. [2] or [1][3].
- If the sources don't contain the answer, say "The sources don't cover this." and nothing else.
- Be concise: at most 5 sentences. Don't mention the sources by any other name than their number."""


class CrossEncoderReranker:
    """Scores (question, chunk) pairs jointly. Slower than retrieval, much sharper."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: List[Hit], k: int) -> List[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -scores[i])[:k]
        return [Hit(hits[i].chunk, float(scores[i]), r) for r, i in enumerate(order)]


def format_sources(hits: List[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h.chunk.title}, p. {h.chunk.page})\n{h.chunk.text}"
                       for i, h in enumerate(hits))


def build_prompt(question: str, hits: List[Hit]) -> str:
    return f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"


def cited_numbers(answer: str) -> List[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


@dataclass
class Answer:
    question: str
    text: str
    sources: List[Hit]
    cited: List[int]
    invalid_citations: List[int] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.text,
            "cited": self.cited,
            "invalid_citations": self.invalid_citations,
            "sources": [{"n": i + 1, "chunk_id": h.chunk.chunk_id, "arxiv_id": h.chunk.arxiv_id,
                         "title": h.chunk.title, "page": h.chunk.page, "score": round(h.score, 4),
                         "text": h.chunk.text} for i, h in enumerate(self.sources)],
            "timings_ms": self.timings,
        }


class RAG:
    def __init__(self, index: Index, llm, reranker: Optional[CrossEncoderReranker] = None,
                 top_k: int = 5, candidates: int = 30):
        self.index = index
        self.llm = llm
        self.reranker = reranker
        self.top_k = top_k
        self.candidates = candidates

    def retrieve(self, question: str, mode: str = "hybrid", rerank: bool = True, k: int = None) -> List[Hit]:
        k = k or self.top_k
        if rerank and self.reranker is not None:
            hits = self.index.search(question, self.candidates, mode)
            return self.reranker.rerank(question, hits, k)
        return self.index.search(question, k, mode)

    def ask(self, question: str, mode: str = "hybrid", rerank: bool = True) -> Answer:
        t0 = time.perf_counter()
        hits = self.retrieve(question, mode, rerank)
        t1 = time.perf_counter()
        text = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, hits))
        t2 = time.perf_counter()
        cited = cited_numbers(text)
        invalid = [n for n in cited if not 1 <= n <= len(hits)]
        return Answer(question, text, hits, cited, invalid,
                      {"retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000)})
