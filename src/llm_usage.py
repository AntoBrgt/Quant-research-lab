"""Track every LLM request (cache hits included) for real cost visibility.

Rows are appended to `data/processed/llm_usage.parquet`. Token counts are the
provider's own reported numbers when available; when a provider doesn't expose
them (or on a cache hit, where no call was made), the field is `None` rather
than an invented estimate. Cost estimation from a token count only happens
here, using the small cost tables in `config.py`, and only when both a known
model and real token counts are present.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import config

COLUMNS = [
    "timestamp",
    "provider",
    "model",
    "operation",
    "cache_hit",
    "input_tokens",
    "output_tokens",
    "estimated_cost",
    "cache_key",
]


def _cost_for(provider: str, model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    if input_tokens is None or output_tokens is None:
        return None

    table = {
        "openai": config.OPENAI_COST_PER_1K_TOKENS,
        "anthropic": config.ANTHROPIC_COST_PER_1K_TOKENS,
    }.get(provider)

    if provider == "ollama":
        return 0.0  # local compute, no per-token billing

    if table is None or model not in table:
        return None

    rates = table[model]
    return round((input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"], 6)


def record_call(
    provider: str,
    model: str,
    operation: str,
    cache_hit: bool,
    cache_key: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    usage_path: Optional[Path] = None,
) -> dict:
    """Append one usage row and return it as a dict.

    `usage_path` defaults to `config.LLM_USAGE_PATH`, read at call time (not as
    a parameter default, which would bind once at import time and ignore any
    later change/monkeypatch of config.LLM_USAGE_PATH).
    """
    usage_path = usage_path or config.LLM_USAGE_PATH
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "operation": operation,
        "cache_hit": bool(cache_hit),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": _cost_for(provider, model, input_tokens, output_tokens),
        "cache_key": cache_key,
    }

    usage_path.parent.mkdir(parents=True, exist_ok=True)

    if usage_path.exists():
        existing = pd.read_parquet(usage_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row], columns=COLUMNS)

    updated.to_parquet(usage_path, index=False)
    return row


def load_usage(usage_path: Optional[Path] = None) -> pd.DataFrame:
    usage_path = usage_path or config.LLM_USAGE_PATH
    if not usage_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(usage_path)


def summarize(usage_path: Optional[Path] = None) -> dict:
    """Compact summary: total calls, cache hits/misses, known cost total."""
    df = load_usage(usage_path)
    if df.empty:
        return {"total_rows": 0, "cache_hits": 0, "llm_calls": 0, "known_cost_total": 0.0}

    return {
        "total_rows": len(df),
        "cache_hits": int(df["cache_hit"].sum()),
        "llm_calls": int((~df["cache_hit"]).sum()),
        "known_cost_total": float(df["estimated_cost"].dropna().sum()),
    }
