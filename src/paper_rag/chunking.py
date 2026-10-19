"""Splitting papers into overlapping chunks.

Chunks follow sentence boundaries and are about `size` words long, with
`overlap` words repeated at the start of the next one so an idea that
straddles a boundary still shows up whole in at least one chunk. Every
chunk remembers its paper and the page it starts on, so answers can cite
"[paper, p. 4]".
"""

import re
from dataclasses import asdict, dataclass
from typing import List

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    title: str
    page: int        # 1-based page where the chunk starts
    text: str

    def to_dict(self):
        return asdict(self)


def split_sentences(text: str) -> List[str]:
    text = " ".join(text.split())
    return [s for s in _SENT.split(text) if s]


def chunk_pages(arxiv_id: str, title: str, pages: List[str],
                size: int = 220, overlap: int = 40) -> List[Chunk]:
    # sentence list with the page each sentence came from
    sents = [(p + 1, s) for p, page in enumerate(pages) for s in split_sentences(page)]
    chunks: List[Chunk] = []
    i = 0
    while i < len(sents):
        words, j = 0, i
        while j < len(sents) and (words < size or j == i):
            words += len(sents[j][1].split())
            j += 1
        text = " ".join(s for _, s in sents[i:j])
        chunks.append(Chunk(f"{arxiv_id}#{len(chunks)}", arxiv_id, title, sents[i][0], text))
        if j >= len(sents):
            break
        # step back far enough to repeat ~overlap words
        back, k = 0, j
        while k > i + 1 and back < overlap:
            k -= 1
            back += len(sents[k][1].split())
        i = k
    return chunks
