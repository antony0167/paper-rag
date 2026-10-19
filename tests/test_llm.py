from paper_rag import llm
from paper_rag.llm import AnthropicLLM, LLMError, NO_ANSWER, USAGE


def fake_post(responses, seen):
    def post(url, headers, payload, **kw):
        seen.append(dict(payload))
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return post


def test_reads_text_after_thinking_and_adds_headroom(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_post", fake_post([{
        "content": [{"type": "thinking", "thinking": "", "signature": "x"},
                    {"type": "text", "text": "Answer [1]."}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}], seen))
    m = AnthropicLLM("claude-haiku-5-5", "key")
    assert m.complete("s", "u", max_tokens=300) == "Answer [1]."
    assert seen[0]["max_tokens"] == 300 + m.headroom
    assert seen[0]["output_config"] == {"effort": "low"}


def test_drops_rejected_parameters_and_flags_empty_reply(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "_post", fake_post([
        LLMError("400: `temperature` is deprecated for this model"),
        LLMError("400: output_config: Extra inputs are not permitted"),
        {"content": [{"type": "thinking", "thinking": "", "signature": "x"}],
         "stop_reason": "max_tokens", "usage": {"input_tokens": 10, "output_tokens": 50}},
    ], seen))
    before = USAGE.incomplete.get("max_tokens", 0)
    m = AnthropicLLM("claude-haiku-5-5", "key")
    out = m.complete("s", "u")
    assert out.startswith(NO_ANSWER) and "max_tokens" in out
    assert "temperature" not in seen[-1] and "output_config" not in seen[-1]
    assert USAGE.incomplete["max_tokens"] == before + 1
