"""Settings, read from environment variables (or a .env file) with sane defaults.

LLM_PROVIDER picks the generator:
    ollama     local model through Ollama's OpenAI-compatible endpoint (default)
    openai     OpenAI API (needs OPENAI_API_KEY)
    anthropic  Anthropic API (needs ANTHROPIC_API_KEY)
    openai_compatible  anything else that speaks /v1/chat/completions (vLLM, LM Studio...)
"""

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0].strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


_DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
    "openai_compatible": "local-model",
}


@dataclass
class Settings:
    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    results_dir: Path = Path("results")

    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_words: int = 220
    chunk_overlap: int = 40

    llm_provider: str = "ollama"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    judge_model: str = ""          # defaults to llm_model

    top_k: int = 5
    candidates: int = 30           # retrieved before reranking
    max_cost_usd: float = 1.0      # hard stop for paid APIs (RAG_MAX_COST_USD)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        e = os.environ.get
        s = cls(
            data_dir=Path(e("RAG_DATA_DIR", "data")),
            results_dir=Path(e("RAG_RESULTS_DIR", "results")),
            embed_model=e("RAG_EMBED_MODEL", cls.embed_model),
            rerank_model=e("RAG_RERANK_MODEL", cls.rerank_model),
            llm_provider=e("LLM_PROVIDER", "ollama").lower(),
            llm_model=e("LLM_MODEL", ""),
            llm_base_url=e("LLM_BASE_URL", ""),
            top_k=int(e("RAG_TOP_K", 5)),
            candidates=int(e("RAG_CANDIDATES", 30)),
            max_cost_usd=float(e("RAG_MAX_COST_USD", 1.0)),
        )
        s.index_dir = s.data_dir / "index"
        s.llm_model = s.llm_model or _DEFAULT_MODELS.get(s.llm_provider, "")
        s.judge_model = e("JUDGE_MODEL", "") or s.llm_model
        if s.llm_provider == "openai":
            s.llm_api_key = e("OPENAI_API_KEY", "")
            s.llm_base_url = s.llm_base_url or "https://api.openai.com/v1"
        elif s.llm_provider == "anthropic":
            s.llm_api_key = e("ANTHROPIC_API_KEY", "")
            s.llm_base_url = s.llm_base_url or "https://api.anthropic.com"
        elif s.llm_provider == "ollama":
            s.llm_base_url = s.llm_base_url or "http://localhost:11434/v1"
        else:
            s.llm_api_key = e("LLM_API_KEY", "")
        return s
