"""Command line.

    paper-rag ingest                    # download + index the papers in papers.txt
    paper-rag ingest --query "retrieval augmented generation" --max 40
    paper-rag ask "How does Self-RAG decide when to retrieve?"
    paper-rag search "lost in the middle"
    paper-rag serve                     # HTTP API on :8000
    paper-rag eval                      # build the test set and run the evaluation
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Settings


def _embedder(s: Settings):
    from .embed import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(s.embed_model)


def load_rag(s: Settings, with_llm=True, with_reranker=True):
    from .index import Index
    from .llm import USAGE, make_llm
    from .pipeline import RAG, CrossEncoderReranker

    USAGE.max_cost_usd = s.max_cost_usd

    if not (s.index_dir / "chunks.jsonl").exists():
        sys.exit("No index yet. Run `paper-rag ingest` first.")
    index = Index.load(s.index_dir, _embedder(s))
    llm = make_llm(s) if with_llm else None
    reranker = CrossEncoderReranker(s.rerank_model) if with_reranker else None
    return RAG(index, llm, reranker, s.top_k, s.candidates)


def cmd_ingest(args, s: Settings):
    from .chunking import chunk_pages
    from .index import Index
    from .papers import download_pdf, fetch_metadata, pdf_pages

    if args.query:
        papers = fetch_metadata(query=args.query, max_results=args.max)
    else:
        ids = [l.split("#")[0].strip() for l in Path(args.ids_file).read_text().splitlines()]
        papers = fetch_metadata(ids=[i for i in ids if i])
    print(f"{len(papers)} papers")

    chunks, kept = [], []
    for p in papers:
        try:
            pdf = download_pdf(p, s.data_dir / "pdfs")
            pages = pdf_pages(pdf)
        except Exception as e:  # one broken PDF shouldn't stop the run
            print(f"  skip {p.arxiv_id}: {e}")
            continue
        cs = chunk_pages(p.arxiv_id, p.title, pages, s.chunk_words, s.chunk_overlap)
        chunks += cs
        kept.append(p.to_dict())
        print(f"  {p.arxiv_id}  {len(pages):>3} pages {len(cs):>4} chunks  {p.title[:70]}")

    t = time.time()
    index = Index.build(chunks, _embedder(s))
    index.save(s.index_dir)
    (s.data_dir / "papers.json").write_text(json.dumps(kept, indent=2))
    print(f"indexed {len(chunks)} chunks from {len(kept)} papers in {time.time() - t:.0f}s -> {s.index_dir}")


def cmd_search(args, s: Settings):
    rag = load_rag(s, with_llm=False, with_reranker=not args.no_rerank)
    for h in rag.retrieve(args.query, args.mode, not args.no_rerank, k=args.k):
        print(f"{h.rank + 1}. [{h.chunk.arxiv_id} p.{h.chunk.page}] {h.chunk.title}\n   {h.chunk.text[:220]}...\n")


def cmd_ask(args, s: Settings):
    rag = load_rag(s, with_reranker=not args.no_rerank)
    ans = rag.ask(args.question, args.mode, not args.no_rerank)
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2))
        return
    print(ans.text, "\n")
    for n in ans.cited:
        if 1 <= n <= len(ans.sources):
            c = ans.sources[n - 1].chunk
            print(f"[{n}] {c.title} (arXiv:{c.arxiv_id}, p. {c.page})")


def cmd_serve(args, s: Settings):
    import uvicorn
    from .api import create_app

    uvicorn.run(create_app(load_rag(s)), host=args.host, port=args.port)


def estimate_eval_cost(s: Settings, n_questions: int, n_answers: int, chunk_words: int):
    """Rough upper bound before spending anything. Token counts are generous on purpose."""
    from .llm import _price

    tok = chunk_words * 1.5                        # ~1.5 tokens per word in papers
    q_in = n_questions * 1.5 * (tok + 350)         # ~1.5 attempts per kept question
    q_out = n_questions * 1.5 * 120
    a_in = n_answers * (s.top_k * tok * 2 + 1500)  # answer + faithfulness both see the sources
    a_out = n_answers * 900
    p_in, p_out = _price(s.llm_model)
    return ((q_in + a_in) * p_in + (q_out + a_out) * p_out) / 1e6


def cmd_eval(args, s: Settings):
    from .llm import USAGE, BudgetExceeded

    paid = s.llm_provider != "ollama"
    if paid:
        n_q = 0 if args.reuse_set else args.questions
        est = estimate_eval_cost(s, n_q, args.answers, s.chunk_words)
        print(f"estimated cost with {s.llm_model}: up to ${est:.2f} (hard cap ${s.max_cost_usd:.2f})")
        if est > s.max_cost_usd:
            sys.exit("Estimate is above the cap. Lower --questions/--answers or raise RAG_MAX_COST_USD.")

    rag = load_rag(s)
    out = s.results_dir
    set_path = out / "eval_set.jsonl"
    try:
        _run_eval(args, s, rag, out, set_path)
    except BudgetExceeded as e:
        sys.exit(f"Stopped: {e}. Spent ${USAGE.cost_usd:.4f}.")
    if paid:
        print(f"total cost: ${USAGE.cost_usd:.4f} ({USAGE.calls} calls, "
              f"{USAGE.input_tokens:,} input + {USAGE.output_tokens:,} output tokens)")


def _run_eval(args, s, rag, out, set_path):
    from .evaluate import answer_metrics, build_eval_set, retrieval_metrics, save_json
    from .llm import USAGE, make_llm
    from .report import write_report

    if args.reuse_set and set_path.exists():
        items = [json.loads(l) for l in set_path.read_text().splitlines()]
        print(f"reusing {len(items)} questions from {set_path}")
    else:
        print(f"building test set with {s.llm_model}...")
        items = build_eval_set(rag.index.chunks, rag.llm, n=args.questions, seed=args.seed)
        out.mkdir(parents=True, exist_ok=True)
        set_path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n")

    print("retrieval...")
    retrieval = retrieval_metrics(rag, items)
    save_json(retrieval, out / "retrieval.json")
    for name, m in retrieval.items():
        print(f"  {name:<16} R@1 {m['recall@1']:.2f}  R@5 {m['recall@5']:.2f}  MRR {m['mrr@10']:.2f}")

    answers = None
    if args.answers > 0:
        print(f"answers ({args.answers} questions, judge: {s.judge_model})...")
        judge = make_llm(s, model=s.judge_model)
        answers = answer_metrics(rag, items[: args.answers], judge)
        save_json(answers, out / "answers.json")
        print(json.dumps(answers["summary"], indent=2))

    meta = {"llm": f"{s.llm_provider}/{s.llm_model}", "judge": s.judge_model,
            "embed_model": s.embed_model, "rerank_model": s.rerank_model,
            "chunks": len(rag.index.chunks),
            "papers": len({c.arxiv_id for c in rag.index.chunks}), "questions": len(items),
            "api_usage": USAGE.summary()}
    save_json(meta, out / "run_info.json")
    write_report(out, retrieval, answers["summary"] if answers else None, meta)
    print(f"report written to {out / 'summary.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="paper-rag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest")
    p.add_argument("--ids-file", default="papers.txt")
    p.add_argument("--query")
    p.add_argument("--max", type=int, default=30)

    for name in ("search", "ask"):
        p = sub.add_parser(name)
        p.add_argument("query" if name == "search" else "question")
        p.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
        p.add_argument("--no-rerank", action="store_true")
        if name == "search":
            p.add_argument("-k", type=int, default=5)
        else:
            p.add_argument("--json", action="store_true")

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("eval")
    p.add_argument("--questions", type=int, default=150)
    p.add_argument("--answers", type=int, default=60, help="0 = retrieval only")
    p.add_argument("--reuse-set", action="store_true", help="reuse results/eval_set.jsonl")
    p.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    s = Settings.from_env()
    {"ingest": cmd_ingest, "search": cmd_search, "ask": cmd_ask,
     "serve": cmd_serve, "eval": cmd_eval}[args.cmd](args, s)


if __name__ == "__main__":
    main()
