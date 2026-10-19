import pytest

from paper_rag.chunking import Chunk
from paper_rag.embed import HashingEmbedder
from paper_rag.index import Index

DOCS = {
    "p1": ("Dense Passage Retrieval",
           ["Dense passage retrieval encodes questions and passages with two BERT encoders. "
            "The similarity is the dot product of the two embeddings. "
            "Training uses in-batch negatives, where the other passages in a batch serve as negatives."]),
    "p2": ("Lost in the Middle",
           ["Language models use information at the beginning and end of a long context best. "
            "Performance drops when the relevant passage sits in the middle of the input. "
            "This U-shaped curve appears even for models built for long contexts."]),
    "p3": ("BM25 baselines",
           ["BM25 is a bag-of-words ranking function based on term frequency and inverse document frequency. "
            "It remains a strong baseline for zero-shot retrieval on the BEIR benchmark."]),
}


@pytest.fixture
def chunks():
    out = []
    for pid, (title, pages) in DOCS.items():
        for i, text in enumerate(pages):
            out.append(Chunk(f"{pid}#{i}", pid, title, i + 1, text))
    return out


@pytest.fixture
def index(chunks):
    return Index.build(chunks, HashingEmbedder())
