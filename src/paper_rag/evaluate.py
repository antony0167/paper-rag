"""Evaluation.

1. Build a test set: sample chunks, have the LLM write a question that the
   chunk answers plus a short reference answer. The chunk it came from is
   the "gold" passage. Questions that copy long phrases from the chunk are
   dropped, otherwise keyword search gets an unfair head start.

2. Retrieval: for each retriever, is the gold chunk in the top 1 / 5 / 10,
   and at which rank (MRR@10)? Also a lenient "right paper in top 5".

3. Answers: for a subset of questions, compare
     - RAG answers (best retriever + LLM)
     - closed-book answers (same LLM, no sources)
   An LLM judge grades correctness against the reference (1-5) and checks
   whether each RAG answer is supported by the sources it was given.
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from .chunking import Chunk
from .pipeline import RAG

QGEN_SYSTEM = "You write evaluation questions for a search system over research papers. Reply with JSON only."
QGEN_USER = """Passage from "{title}":
\"\"\"
{text}
\"\"\"

Write one question that a researcher could ask and that this passage answers on its own.
- Be specific: about a method, number, finding or definition in the passage, not "what does this paper propose?".
- Paraphrase. Don't copy more than three consecutive words from the passage.
- The question must make sense without seeing the passage (name the method or paper topic if needed).
Then give the answer in one sentence.

If the passage is unusable (reference list, table of numbers without context, author list, broken text), reply {{"skip": true}}.
Otherwise reply {{"question": "...", "answer": "..."}}"""

CLOSED_BOOK_SYSTEM = ("Answer the question from your own knowledge in at most 5 sentences. "
                      "If you are not sure, say you don't know.")

JUDGE_SYSTEM = "You grade answers. Reply with JSON only."
CORRECTNESS_USER = """Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

How well does the candidate answer the question, compared with the reference?
5 = fully correct, 4 = correct with minor gaps, 3 = partly correct, 2 = mostly wrong or vague, 1 = wrong or no answer.
Reply {{"score": <1-5>, "reason": "<at most 15 words>"}}"""

FAITHFUL_USER = """Sources:
{sources}

Answer: {answer}

Is every factual claim in the answer supported by the sources above?
Reply {{"verdict": "supported" | "partial" | "unsupported", "reason": "<at most 15 words>"}}
(An answer that only says the sources don't cover the question counts as "supported".)"""


def parse_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_field(text: str, key: str):
    """Read one field from the judge's reply, even if the JSON was cut off.

    The judge sometimes writes a long "reason" and hits max_tokens before the
    closing brace; the field we care about comes first, so it is still there.
    """
    out = parse_json(text)
    if out and key in out:
        return out[key]
    m = re.search(rf'"{key}"\s*:\s*"?([A-Za-z0-9]+)', text)
    return m.group(1) if m else None


def ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def copies_passage(question: str, passage: str, n: int = 5) -> bool:
    q = re.findall(r"[a-z0-9]+", question.lower())
    p = re.findall(r"[a-z0-9]+", passage.lower())
    return bool(ngrams(q, n) & ngrams(p, n))


def usable_chunk(c: Chunk) -> bool:
    words = c.text.split()
    if len(words) < 80:
        return False
    alpha = sum(ch.isalpha() for ch in c.text) / max(1, len(c.text))
    return alpha > 0.6


# ------------------------------------------------------------------ test set

def build_eval_set(chunks: List[Chunk], llm, n: int = 150, seed: int = 0, log=print) -> List[dict]:
    rng = random.Random(seed)
    by_paper: Dict[str, List[Chunk]] = {}
    for c in chunks:
        if usable_chunk(c):
            by_paper.setdefault(c.arxiv_id, []).append(c)
    for lst in by_paper.values():
        rng.shuffle(lst)
    # round-robin over papers so long papers don't dominate
    queue: List[Chunk] = []
    papers = list(by_paper)
    rng.shuffle(papers)
    pos = {p: 0 for p in papers}
    while any(pos[p] < len(by_paper[p]) for p in papers):
        for p in papers:
            if pos[p] < len(by_paper[p]):
                queue.append(by_paper[p][pos[p]])
                pos[p] += 1

    items, skipped = [], {"unusable": 0, "copied": 0, "bad_json": 0}
    for c in queue:
        if len(items) >= n:
            break
        out = parse_json(llm.complete(QGEN_SYSTEM, QGEN_USER.format(title=c.title, text=c.text),
                                      max_tokens=300, temperature=0.3))
        if not out or "question" not in out:
            skipped["unusable" if out and out.get("skip") else "bad_json"] += 1
            continue
        if copies_passage(out["question"], c.text):
            skipped["copied"] += 1
            continue
        items.append({"id": len(items), "question": out["question"].strip(),
                      "reference": out.get("answer", "").strip(),
                      "gold_chunk": c.chunk_id, "gold_paper": c.arxiv_id})
        if len(items) % 10 == 0:
            log(f"  {len(items)}/{n} questions")
    log(f"built {len(items)} questions (skipped: {skipped})")
    return items


# ----------------------------------------------------------------- retrieval

RETRIEVERS = {
    "bm25": dict(mode="bm25", rerank=False),
    "dense": dict(mode="dense", rerank=False),
    "hybrid": dict(mode="hybrid", rerank=False),
    "hybrid + rerank": dict(mode="hybrid", rerank=True),
}


def retrieval_metrics(rag: RAG, items: List[dict], configs=None, k: int = 10) -> Dict[str, dict]:
    configs = configs or RETRIEVERS
    results = {}
    for name, cfg in configs.items():
        if cfg["rerank"] and rag.reranker is None:
            continue
        hit1 = hit5 = hit10 = paper5 = 0
        rr, ms = 0.0, []
        for it in items:
            t = time.perf_counter()
            hits = rag.retrieve(it["question"], mode=cfg["mode"], rerank=cfg["rerank"], k=k)
            ms.append((time.perf_counter() - t) * 1000)
            ids = [h.chunk.chunk_id for h in hits]
            if it["gold_chunk"] in ids:
                r = ids.index(it["gold_chunk"]) + 1
                rr += 1 / r
                hit1 += r <= 1
                hit5 += r <= 5
                hit10 += r <= 10
            paper5 += it["gold_paper"] in [h.chunk.arxiv_id for h in hits[:5]]
        n = len(items)
        ms.sort()
        results[name] = {
            "recall@1": hit1 / n, "recall@5": hit5 / n, "recall@10": hit10 / n,
            "mrr@10": rr / n, "paper@5": paper5 / n,
            "latency_ms_median": ms[len(ms) // 2],
        }
    return results


# ------------------------------------------------------------------- answers

def answer_metrics(rag: RAG, items: List[dict], judge, mode="hybrid", rerank=True, log=print) -> dict:
    rows = []
    for i, it in enumerate(items):
        ans = rag.ask(it["question"], mode=mode, rerank=rerank)
        closed = rag.llm.complete(CLOSED_BOOK_SYSTEM, it["question"])

        def grade(candidate):
            raw = judge.complete(JUDGE_SYSTEM, CORRECTNESS_USER.format(
                question=it["question"], reference=it["reference"], candidate=candidate), max_tokens=400)
            try:
                score = int(judge_field(raw, "score"))
            except (TypeError, ValueError):
                return None
            return score if 1 <= score <= 5 else None

        verdict = judge_field(judge.complete(JUDGE_SYSTEM, FAITHFUL_USER.format(
            sources="\n\n".join(f"[{j + 1}] {h.chunk.text}" for j, h in enumerate(ans.sources)),
            answer=ans.text), max_tokens=400), "verdict")
        verdict = verdict.lower() if isinstance(verdict, str) else None
        if verdict not in ("supported", "partial", "unsupported"):
            verdict = None
        gold_n = [j + 1 for j, h in enumerate(ans.sources) if h.chunk.chunk_id == it["gold_chunk"]]
        rows.append({
            "id": it["id"], "question": it["question"], "reference": it["reference"],
            "rag_answer": ans.text, "closed_book_answer": closed,
            "rag_score": grade(ans.text), "closed_book_score": grade(closed),
            "faithfulness": verdict,
            "abstained": "don't cover" in ans.text.lower(),
            "cited": ans.cited, "invalid_citations": ans.invalid_citations,
            "cites_gold": bool(set(ans.cited) & set(gold_n)),
        })
        if (i + 1) % 5 == 0:
            log(f"  answered {i + 1}/{len(items)}")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    n = len(rows)
    judged = [r for r in rows if r["faithfulness"] is not None]
    nj = max(1, len(judged))
    summary = {
        "n": n,
        "rag_correctness": mean(r["rag_score"] for r in rows),
        "closed_book_correctness": mean(r["closed_book_score"] for r in rows),
        # faithfulness shares are over answers the judge actually graded
        "faithful_supported": sum(r["faithfulness"] == "supported" for r in judged) / nj,
        "faithful_partial": sum(r["faithfulness"] == "partial" for r in judged) / nj,
        "faithful_unsupported": sum(r["faithfulness"] == "unsupported" for r in judged) / nj,
        "judge_failures": sum(r["rag_score"] is None or r["closed_book_score"] is None
                              or r["faithfulness"] is None for r in rows),
        "abstention_rate": sum(r["abstained"] for r in rows) / n,
        "invalid_citation_rate": sum(bool(r["invalid_citations"]) for r in rows) / n,
        "cites_gold_rate": sum(r["cites_gold"] for r in rows) / n,
    }
    return {"summary": summary, "rows": rows}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
