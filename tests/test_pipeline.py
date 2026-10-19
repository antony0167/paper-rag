import json

import pytest
from fastapi.testclient import TestClient

from paper_rag import llm as llm_mod
from paper_rag.api import create_app
from paper_rag.evaluate import (answer_metrics, build_eval_set, copies_passage, parse_json,
                                retrieval_metrics)
from paper_rag.llm import AnthropicLLM, FakeLLM, OpenAICompatibleLLM
from paper_rag.pipeline import RAG, build_prompt, cited_numbers


def test_prompt_numbers_sources(index):
    hits = index.search("dense retrieval", k=2)
    p = build_prompt("What is DPR?", hits)
    assert "[1] (" in p and "[2] (" in p and p.endswith("Question: What is DPR?")


def test_citations_are_parsed_and_checked(index):
    rag = RAG(index, FakeLLM(lambda s, u: "DPR uses two encoders [1]. Also [9]."), top_k=3)
    ans = rag.ask("How does DPR encode passages?")
    assert ans.cited == [1, 9]
    assert ans.invalid_citations == [9]
    assert cited_numbers("a [2][3] b [2]") == [2, 3]


def test_parse_json_tolerates_chatter():
    assert parse_json('Sure! {"score": 4, "reason": "ok"} hope that helps') == {"score": 4, "reason": "ok"}
    assert parse_json("no json here") is None


def test_copy_filter():
    passage = "Training uses in-batch negatives where other passages serve as negatives"
    assert copies_passage("why do other passages serve as negatives here", passage)
    assert not copies_passage("How are negative examples chosen during training?", passage)


def _qgen(system, user):
    if "Lost in the Middle" in user:
        # copies the passage word for word -> must be filtered out
        return json.dumps({"question": "performance drops when the relevant passage sits in the middle?",
                           "answer": "yes"})
    if "BM25" in user:
        return '{"skip": true}'
    return json.dumps({"question": "How are negatives chosen when training DPR?",
                       "answer": "Other passages in the same batch are used as negatives."})


def test_build_eval_set_filters_and_records_gold(chunks):
    long = [c.__class__(c.chunk_id, c.arxiv_id, c.title, c.page, (c.text + " ") * 6) for c in chunks]
    items = build_eval_set(long, FakeLLM(_qgen), n=10, log=lambda *a: None)
    assert len(items) == 1
    assert items[0]["gold_chunk"] == "p1#0"


def test_retrieval_metrics(index):
    items = [{"id": 0, "question": "in-batch negatives dual encoder training", "reference": "",
              "gold_chunk": "p1#0", "gold_paper": "p1"},
             {"id": 1, "question": "BM25 term frequency baseline BEIR", "reference": "",
              "gold_chunk": "p3#0", "gold_paper": "p3"}]
    m = retrieval_metrics(RAG(index, None), items)
    assert set(m) == {"bm25", "dense", "hybrid"}  # no reranker loaded -> skipped
    assert m["hybrid"]["recall@1"] == 1.0 and m["hybrid"]["mrr@10"] == 1.0


def test_answer_metrics(index):
    def fake(system, user):
        if "grade" in system.lower() and "Reference answer" in user:
            return '{"score": 5, "reason": "x"}'
        if "grade" in system.lower():
            return '{"verdict": "supported", "reason": "x"}'
        if "own knowledge" in system:
            return "I don't know."
        return "Other in-batch passages are negatives [1]."

    llm = FakeLLM(fake)
    items = [{"id": 0, "question": "How are negatives chosen in DPR training?",
              "reference": "In-batch negatives.", "gold_chunk": "p1#0", "gold_paper": "p1"}]
    res = answer_metrics(RAG(index, llm, top_k=3), items, judge=llm, rerank=False, log=lambda *a: None)
    s = res["summary"]
    assert s["rag_correctness"] == 5 and s["faithful_supported"] == 1.0
    assert s["cites_gold_rate"] == 1.0 and s["invalid_citation_rate"] == 0.0


def test_api(index):
    client = TestClient(create_app(RAG(index, FakeLLM(lambda s, u: "Answer [1]."), top_k=2)))
    assert client.get("/health").json()["chunks"] == 3
    r = client.post("/search", json={"query": "BM25 baseline", "k": 2, "rerank": False})
    assert r.status_code == 200 and r.json()[0]["arxiv_id"] == "p3"
    r = client.post("/ask", json={"question": "What is BM25?", "rerank": False})
    assert r.json()["answer"] == "Answer [1]." and r.json()["cited"] == [1]
    assert client.post("/search", json={"query": "x"}).status_code == 422


class _Resp:
    def __init__(self, payload, code=200):
        self.payload, self.status_code, self.text = payload, code, json.dumps(payload)

    def json(self):
        return self.payload


def test_openai_compatible_payload(monkeypatch):
    seen = {}

    def post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return _Resp({"choices": [{"message": {"content": " hi "}}]})

    monkeypatch.setattr(llm_mod.requests, "post", post)
    out = OpenAICompatibleLLM("http://localhost:11434/v1/", "qwen2.5:7b").complete("sys", "user")
    assert out == "hi"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert "Authorization" not in seen["headers"]


def test_anthropic_payload(monkeypatch):
    seen = {}

    def post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return _Resp({"content": [{"type": "text", "text": "hello"}]})

    monkeypatch.setattr(llm_mod.requests, "post", post)
    out = AnthropicLLM("some-model", "key").complete("sys", "user")
    assert out == "hello"
    assert seen["url"].endswith("/v1/messages")
    assert seen["body"]["system"] == "sys" and seen["headers"]["x-api-key"] == "key"


def test_retries_then_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: calls.append(1) or _Resp({}, 503))
    with pytest.raises(llm_mod.LLMError):
        OpenAICompatibleLLM("http://x/v1", "m").complete("s", "u")
    assert len(calls) == 3


def test_cost_cap_stops_paid_calls(monkeypatch):
    usage = llm_mod.Usage(max_cost_usd=0.001)
    monkeypatch.setattr(llm_mod, "USAGE", usage)
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: _Resp(
        {"content": [{"type": "text", "text": "x"}], "usage": {"input_tokens": 5000, "output_tokens": 500}}))
    client = AnthropicLLM("claude-haiku-5-5", "key")
    client.complete("s", "u")           # 5000*0.10/1e6 + 500*0.50/1e6 = $0.00075
    client.complete("s", "u")           # now $0.0015, over the cap
    with pytest.raises(llm_mod.BudgetExceeded):
        client.complete("s", "u")
    assert usage.calls == 2 and usage.cost_usd == pytest.approx(0.0015)


def test_local_models_are_free(monkeypatch):
    usage = llm_mod.Usage(max_cost_usd=0.0001)
    monkeypatch.setattr(llm_mod, "USAGE", usage)
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: _Resp(
        {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 10**6, "completion_tokens": 10**6}}))
    client = OpenAICompatibleLLM("http://localhost:11434/v1", "qwen2.5:7b", free=True)
    for _ in range(3):
        client.complete("s", "u")
    assert usage.cost_usd == 0.0


def test_anthropic_retries_without_temperature(monkeypatch):
    bodies = []

    def post(url, headers, json, timeout):
        bodies.append(dict(json))
        if "temperature" in json:
            return _Resp({"type": "error", "error": {"message": "`temperature` is deprecated for this model."}}, 400)
        return _Resp({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}})

    monkeypatch.setattr(llm_mod.requests, "post", post)
    client = AnthropicLLM("claude-haiku-5-5", "key")
    assert client.complete("s", "u") == "ok"
    assert client.complete("s", "u") == "ok"      # remembers: no failing request the second time
    assert ["temperature" in b for b in bodies] == [True, False, False]
