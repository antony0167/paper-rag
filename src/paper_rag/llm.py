"""LLM clients. Plain HTTP, no vendor SDKs.

`OpenAICompatibleLLM` covers OpenAI, Ollama, vLLM and LM Studio, which all
serve /v1/chat/completions. `AnthropicLLM` talks to the Messages API.
"""

import os
import time
from typing import Callable, Optional

import requests


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


# USD per million tokens (input, output), from Anthropic's pricing page
# (October 2026, prompts under 100k tokens). For any other paid model set
# RAG_PRICE_IN / RAG_PRICE_OUT, otherwise the pessimistic fallback is used.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
# unknown paid models are costed pessimistically so the cap still protects you
FALLBACK_PRICE = (5.00, 25.00)


class Usage:
    """Counts tokens across every call and stops once the cost cap is reached.

    The cap is checked before each request, so the most it can be exceeded by
    is the cost of a single call (a fraction of a cent with small models).
    """

    def __init__(self, max_cost_usd: Optional[float] = None):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0
        self.max_cost_usd = max_cost_usd
        self.incomplete = {}  # stop_reason -> count of cut-off or refused replies

    def check(self):
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"cost cap reached: ${self.cost_usd:.4f} >= ${self.max_cost_usd:.2f} "
                                 f"(raise RAG_MAX_COST_USD to continue)")

    def add(self, model: str, input_tokens: int, output_tokens: int, free: bool = False):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if not free:
            p_in, p_out = _price(model)
            self.cost_usd += (input_tokens * p_in + output_tokens * p_out) / 1e6

    def summary(self):
        out = {"calls": self.calls, "input_tokens": self.input_tokens,
               "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 4)}
        if self.incomplete:
            out["incomplete_replies"] = dict(self.incomplete)
        return out


def _price(model: str):
    if os.environ.get("RAG_PRICE_IN") and os.environ.get("RAG_PRICE_OUT"):
        return float(os.environ["RAG_PRICE_IN"]), float(os.environ["RAG_PRICE_OUT"])
    for name, price in PRICES.items():
        if model.startswith(name):
            return price
    return FALLBACK_PRICE


# shared by every client in the process, so the cap covers the whole run
USAGE = Usage()


def _post(url, headers, payload, retries=3, timeout=180):
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.ConnectionError as e:
            if attempt == retries - 1:
                raise LLMError(f"could not reach {url}: is the server running?") from e
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 529) and attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))
            continue
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code} from {url}: {r.text[:300]}")
        return r.json()
    raise LLMError("unreachable")


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", free: bool = False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.free = free  # local models (Ollama) cost nothing
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def complete(self, system: str, user: str, max_tokens: int = 600, temperature: float = 0.0) -> str:
        USAGE.check()
        data = _post(self.url, self.headers, {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        u = data.get("usage") or {}
        USAGE.add(self.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), self.free)
        return data["choices"][0]["message"]["content"].strip()


NO_ANSWER = "[no answer from the model]"


class AnthropicLLM:
    """Messages API client.

    Newer Claude models think before answering by default, and the hidden
    thinking tokens count against max_tokens. A tight limit can then end the
    reply mid-sentence or before any text at all. So callers keep passing the
    length they want for the *answer*, and this client adds headroom for the
    thinking and asks for low effort, which keeps thinking (and cost) small.
    """

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com"):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Put a line `ANTHROPIC_API_KEY=sk-ant-...` "
                           "in .env (no # in front), in the folder you run paper-rag from.")
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.model = model
        self.headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                        "content-type": "application/json"}
        self.headroom = int(os.environ.get("RAG_THINKING_HEADROOM", 2000))
        effort = os.environ.get("RAG_EFFORT", "low")
        # optional parameters some models reject; each is dropped after the first 400 that names it
        self.optional = {"temperature": None, "output_config": {"effort": effort} if effort else None}

    def complete(self, system: str, user: str, max_tokens: int = 600, temperature: float = 0.0) -> str:
        USAGE.check()
        payload = {"model": self.model, "system": system, "max_tokens": max_tokens + self.headroom,
                   "messages": [{"role": "user", "content": user}]}
        for key, value in self.optional.items():
            if key == "temperature":
                payload[key] = temperature
            elif value is not None:
                payload[key] = value
        while True:
            try:
                data = _post(self.url, self.headers, payload)
                break
            except LLMError as e:
                msg = str(e)
                rejected = [k for k in self.optional if k in payload and
                            (k in msg or (k == "output_config" and "effort" in msg))]
                if not rejected:
                    raise
                for k in rejected:
                    payload.pop(k)
                    self.optional.pop(k)
        u = data.get("usage") or {}
        USAGE.add(self.model, u.get("input_tokens", 0), u.get("output_tokens", 0))
        stop = data.get("stop_reason")
        # thinking blocks come first, so pick text blocks by type, not position
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text").strip()
        if stop in ("max_tokens", "refusal") or not text:
            USAGE.incomplete[stop or "empty"] = USAGE.incomplete.get(stop or "empty", 0) + 1
        if not text:
            return f"{NO_ANSWER} (stop_reason: {stop})"
        return text


class FakeLLM:
    """For tests: returns whatever the given function produces."""

    def __init__(self, fn: Optional[Callable[[str, str], str]] = None):
        self.fn = fn or (lambda system, user: "I don't know.")
        self.calls = []

    def complete(self, system, user, max_tokens=600, temperature=0.0):
        self.calls.append((system, user))
        return self.fn(system, user)


def make_llm(settings, model: Optional[str] = None):
    model = model or settings.llm_model
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(model, settings.llm_api_key, settings.llm_base_url)
    if settings.llm_provider == "openai" and not settings.llm_api_key:
        raise LLMError("OPENAI_API_KEY is not set")
    return OpenAICompatibleLLM(settings.llm_base_url, model, settings.llm_api_key,
                               free=settings.llm_provider == "ollama")
