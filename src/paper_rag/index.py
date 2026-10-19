"""The search index: BM25 for keywords, embeddings for meaning, and a hybrid of both.

Hybrid search uses reciprocal rank fusion (RRF): each retriever ranks the
chunks, and a chunk scores sum(1 / (k + rank)) over the lists it appears in.
It only looks at ranks, so the two very different score scales (BM25
counts vs cosine similarity) never have to be calibrated against each other.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

_STOP = set("""a an the of and or to in on for with by from as at is are was were be been this that
these those it its we our they their which who what when where how than then into over also can
using use used based not but such via""".split())


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def rrf(rankings: List[List[int]], k: int = 60) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    return scores


class Index:
    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray, embedder):
        self.chunks = chunks
        self.emb = embeddings
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.title + " " + c.text) for c in chunks])

    # --------------------------------------------------------------- build/io
    @classmethod
    def build(cls, chunks: List[Chunk], embedder) -> "Index":
        texts = [f"{c.title}\n{c.text}" for c in chunks]
        return cls(chunks, embedder.embed_documents(texts), embedder)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict()) + "\n")
        np.save(folder / "embeddings.npy", self.emb)

    @classmethod
    def load(cls, folder: Path, embedder) -> "Index":
        with open(folder / "chunks.jsonl") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, np.load(folder / "embeddings.npy"), embedder)

    # ----------------------------------------------------------------- search
    def _bm25_rank(self, query: str, n: int) -> List[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def _dense_rank(self, query: str, n: int) -> List[int]:
        q = self.embedder.embed_query(query)
        sims = self.emb @ q
        return list(np.argsort(-sims)[:n])

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> List[Hit]:
        if mode == "bm25":
            order = self._bm25_rank(query, k)
            scores = {i: 1.0 / (r + 1) for r, i in enumerate(order)}
        elif mode == "dense":
            order = self._dense_rank(query, k)
            scores = {i: 1.0 / (r + 1) for r, i in enumerate(order)}
        elif mode == "hybrid":
            pool = max(k * 4, 50)
            scores = rrf([self._bm25_rank(query, pool), self._dense_rank(query, pool)])
            order = sorted(scores, key=scores.get, reverse=True)[:k]
        else:
            raise ValueError(f"unknown mode {mode}")
        return [Hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(order)]
