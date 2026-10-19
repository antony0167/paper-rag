"""Text embeddings.

Real runs use a sentence-transformers model (default: bge-small, 384-d,
fast on CPU). Tests use HashingEmbedder, which needs no download and is
deterministic.
"""

import hashlib
import re
from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: List[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    # bge and e5 models were trained with an instruction prefix on queries
    QUERY_PREFIX = {
        "bge": "Represent this sentence for searching relevant passages: ",
        "e5": "query: ",
    }
    DOC_PREFIX = {"e5": "passage: "}

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        family = next((k for k in ("bge", "e5") if k in model_name.lower()), None)
        self.q_prefix = self.QUERY_PREFIX.get(family, "")
        self.d_prefix = self.DOC_PREFIX.get(family, "")

    def embed_documents(self, texts):
        return self.model.encode([self.d_prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 100,
                                 convert_to_numpy=True).astype("float32")

    def embed_query(self, text):
        return self.model.encode([self.q_prefix + text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype("float32")


class HashingEmbedder:
    """Bag of hashed words + bigrams. Only for tests and smoke runs."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        toks = re.findall(r"[a-z0-9]+", text.lower())
        for t in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")

    def embed_query(self, text):
        return self._vec(text)
