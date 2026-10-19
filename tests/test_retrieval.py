from paper_rag.chunking import chunk_pages, split_sentences
from paper_rag.embed import HashingEmbedder
from paper_rag.index import Index, rrf, tokenize
from paper_rag.papers import clean_page, parse_atom


def test_chunks_respect_size_and_overlap():
    sent = "Every sentence in this test has eight words."
    pages = [" ".join([sent] * 30), " ".join([sent] * 30)]
    chunks = chunk_pages("x", "T", pages, size=50, overlap=14)
    assert len(chunks) > 5
    for c in chunks[:-1]:
        n = len(c.text.split())
        assert 50 <= n < 50 + 8          # stops right after crossing the size
    # consecutive chunks share whole sentences: 14 words of overlap -> 2 sentences
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert a[-16:] == b[:16]
    assert chunks[0].page == 1 and chunks[-1].page == 2
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_sentence_split_keeps_numbers():
    s = split_sentences("We use k = 5. Results improve by 3.2 points. (See Table 2.) Done.")
    assert s[1] == "Results improve by 3.2 points."


def test_clean_page_rejoins_hyphenation():
    assert clean_page("retrie-\nval works") == "retrieval works"


def test_tokenize_drops_stopwords():
    assert tokenize("The retrieval of the in-batch negatives") == ["retrieval", "in-batch", "negatives"]


def test_rrf_prefers_items_ranked_well_by_both():
    scores = rrf([[1, 2, 3], [3, 1, 2]])
    assert max(scores, key=scores.get) == 1


def test_search_modes_find_the_right_chunk(index):
    for mode in ("bm25", "dense", "hybrid"):
        hits = index.search("in-batch negatives for training the dual encoder", k=2, mode=mode)
        assert hits[0].chunk.arxiv_id == "p1", mode
    hits = index.search("relevant passage in the middle of a long context", k=1, mode="hybrid")
    assert hits[0].chunk.arxiv_id == "p2"


def test_index_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = Index.load(tmp_path, HashingEmbedder())
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    q = "BM25 zero-shot baseline"
    assert [h.chunk.chunk_id for h in loaded.search(q, 3)] == [h.chunk.chunk_id for h in index.search(q, 3)]


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v4</id>
    <published>2020-05-22T17:59:32Z</published>
    <title>Retrieval-Augmented Generation for
      Knowledge-Intensive NLP Tasks</title>
    <summary>  Large pre-trained language models have been shown to store factual knowledge.  </summary>
    <author><name>Patrick Lewis</name></author>
    <author><name>Ethan Perez</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2005.11401v4" rel="related" type="application/pdf"/>
  </entry>
</feed>"""


def test_parse_atom():
    [p] = parse_atom(ATOM)
    assert p.arxiv_id == "2005.11401"
    assert p.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert p.authors == ["Patrick Lewis", "Ethan Perez"]
    assert p.published == "2020-05-22"
    assert p.pdf_url.endswith("2005.11401v4")


def test_dotenv_parsing(tmp_path, monkeypatch):
    from paper_rag.config import _load_dotenv
    for k in ("A_KEY", "B_KEY", "C_KEY", "D_KEY"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text('export A_KEY=sk-1\nB_KEY = "sk-2"   # comment\n# C_KEY=no\nD_KEY=\n')
    _load_dotenv(env)
    import os
    assert os.environ["A_KEY"] == "sk-1" and os.environ["B_KEY"] == "sk-2"
    assert "C_KEY" not in os.environ and "D_KEY" not in os.environ
