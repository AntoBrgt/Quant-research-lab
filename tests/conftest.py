"""Shared pytest fixtures.

Tests import `src/` modules as flat top-level modules (matching how the
scripts import each other, since this project runs as plain scripts, not an
installed package) -- so `src/` is put on `sys.path` here, once, for the whole
test session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))  # so `import app` works (app.py lives at repo root, not src/)

import cache  # noqa: E402
import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own empty cache dir and a fresh usage log path."""
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "LLM_USAGE_PATH", tmp_path / "llm_usage.parquet")
    yield


@pytest.fixture
def synthetic_documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL", "filing_type": "10-K", "filing_date": "2026-01-01",
                "section": "Risk Factors", "chunk_id": "AAPL_10K_RISK_FACTORS_000",
                "chunk_index": 0, "text": "Competition in the smartphone market is intense.",
                "source_file": "AAPL.txt",
            },
            {
                "ticker": "MSFT", "filing_type": "10-K", "filing_date": "2026-01-01",
                "section": "Risk Factors", "chunk_id": "MSFT_10K_RISK_FACTORS_000",
                "chunk_index": 0, "text": "Cloud infrastructure competition is significant.",
                "source_file": "MSFT.txt",
            },
        ]
    )


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {
            "adj_close": 100 + pd.Series(range(300)).astype(float) * 0.1,
            "volume": [1_000_000] * 300,
        },
        index=dates,
    )
    prices.index.name = "date"
    return prices


class FakeSignal:
    """Minimal stand-in for signal_extraction.Signal, avoiding a pydantic import in tests."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class FakeLLM:
    """Fake chat model: with_structured_output(...).invoke(...) returns canned signals.

    Tracks call_count so tests can assert the LLM was (or wasn't) actually called.
    """

    def __init__(self, signals: list[dict]):
        self._signals = signals
        self.call_count = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, _prompt):
        self.call_count += 1
        parsed = _FakeExtraction(self._signals)
        return {"raw": None, "parsed": parsed, "parsing_error": None}


class _FakeExtraction:
    def __init__(self, signal_dicts: list[dict]):
        self.signals = [FakeSignal(**d) for d in signal_dicts]
