"""Central configuration for quant-research-lab.

Every other module reads paths, provider settings, and version constants from
here instead of redefining them. `.env` is loaded before any value below is
evaluated, so environment overrides always take effect.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

DOCUMENTS_PATH = PROCESSED_DATA_DIR / "documents.parquet"
SIGNALS_PATH = PROCESSED_DATA_DIR / "signals.parquet"
NEWS_SIGNALS_PATH = PROCESSED_DATA_DIR / "news_signals.parquet"
LLM_USAGE_PATH = PROCESSED_DATA_DIR / "llm_usage.parquet"
PRICES_DIR = RAW_DATA_DIR / "prices"

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.getenv(
    "OLLAMA_MODEL" if LLM_PROVIDER == "ollama" else
    "ANTHROPIC_MODEL" if LLM_PROVIDER == "anthropic" else
    "OPENAI_MODEL",
    {"ollama": "llama3.2", "anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini"}.get(
        LLM_PROVIDER, "llama3.2"
    ),
)
LLM_ENABLED = _env_bool("LLM_ENABLED", True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Cache / versioning
# ---------------------------------------------------------------------------

CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)
PROMPT_VERSION = "v1"
SCHEMA_VERSION = "v1"

# ---------------------------------------------------------------------------
# Cost / safety guards
# ---------------------------------------------------------------------------

MAX_LLM_INPUT_CHARS = _env_int("MAX_LLM_INPUT_CHARS", 8000)
MAX_LLM_CALLS_PER_RUN = _env_int("MAX_LLM_CALLS_PER_RUN", 500)
# A well-formed chunk should produce a handful of signals. A local model that
# gets stuck in a repetition loop can otherwise emit hundreds from one chunk
# (observed: 972 and 909 "signals" from single 4500-char chunks).
MAX_SIGNALS_PER_CHUNK = _env_int("MAX_SIGNALS_PER_CHUNK", 15)
# A stuck/hanging request (observed: multi-hour hangs on dense chunks) is worse
# than a failed one -- fail fast and move on instead of blocking the whole run.
LLM_REQUEST_TIMEOUT_SECONDS = _env_int("LLM_REQUEST_TIMEOUT_SECONDS", 120)

# Rough, clearly-labeled cost table for dry-run estimation only. Real usage
# (llm_usage.py) never estimates -- it records provider-reported tokens or None.
OPENAI_COST_PER_1K_TOKENS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}
ANTHROPIC_COST_PER_1K_TOKENS = {
    "claude-haiku-4-5": {"input": 0.001, "output": 0.005},
    "claude-sonnet-5": {"input": 0.002, "output": 0.01},
    "claude-opus-5": {"input": 0.005, "output": 0.025},
}
