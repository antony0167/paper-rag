"""End-to-end run of the CLI with network, models and LLM replaced by stand-ins."""

import json

from paper_rag import cli, llm as llm_mod, papers as papers_mod
from paper_rag.embed import HashingEmbedder
from paper_rag.llm import FakeLLM
from paper_rag.papers import Paper

from conftest import DOCS


def _fake_llm(system, user):
    if "evaluation questions" in system:
        if "in-batch" in user:
            return '{"question": "How does DPR pick negative examples during training?", "answer": "From the same batch."}'
        if "U-shaped" in user:
            return '{"question": "Where in a long prompt do models use information worst?", "answer": "The middle."}'
        return '{"skip": true}'
    if "grade" in system.lower():
        return '{"score": 4, "verdict": "supported", "reason": "ok"}'
    if "own knowledge" in system:
        return "Not sure."
    return "Answer based on the sources [1]."


def test_ingest_ask_eval(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    papers = [Paper(pid, title, ["A"], "2020-01-01", "", "") for pid, (title, _) in DOCS.items()]
    monkeypatch.setattr(papers_mod, "fetch_metadata", lambda ids=None, query=None, max_results=30: papers)
    monkeypatch.setattr(papers_mod, "download_pdf", lambda p, folder: p.arxiv_id)
    # make each page long enough to pass the "usable chunk" filter
    monkeypatch.setattr(papers_mod, "pdf_pages", lambda pid: [" ".join(DOCS[pid][1] * 6)])
    monkeypatch.setattr(cli, "_embedder", lambda s: HashingEmbedder())
    monkeypatch.setattr(llm_mod, "make_llm", lambda s, model=None: FakeLLM(_fake_llm))
    monkeypatch.setattr("paper_rag.pipeline.CrossEncoderReranker", None)

    (tmp_path / "papers.txt").write_text("p1\np2  # comment\n# skipped line\np3\n")
    cli.main(["ingest"])
    assert (tmp_path / "data/index/chunks.jsonl").exists()

    def no_reranker(s, with_llm=True, with_reranker=True):
        return orig_load(s, with_llm, with_reranker=False)
    orig_load = cli.load_rag
    monkeypatch.setattr(cli, "load_rag", no_reranker)

    cli.main(["ask", "How are negatives chosen in DPR?", "--json"])
    out = capsys.readouterr().out
    ans = json.loads(out[out.index("{"):])
    assert ans["cited"] == [1] and ans["sources"]

    cli.main(["eval", "--questions", "5", "--answers", "2"])
    res = tmp_path / "results"
    assert len((res / "eval_set.jsonl").read_text().splitlines()) == 2
    retrieval = json.loads((res / "retrieval.json").read_text())
    assert set(retrieval) == {"bm25", "dense", "hybrid"}
    assert (res / "summary.md").exists() and (res / "retrieval.png").exists()
    info = json.loads((res / "run_info.json").read_text())
    assert info["papers"] == 3
