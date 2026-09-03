"""Aggregate cached, already-computed signals into one company research object.

This module never calls the LLM. It only reads what `extract_signals.py` has
already produced (SEC signals, news signals) and what `market_features.py`
computes from price history, and combines them into a single per-ticker
research object. This is the "shared research cache" read path: any number of
portfolios can call `load_company_research("AAPL")` without triggering any new
LLM work, which is the entire point of the cache-first architecture.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import config
import market_features
import price_provider


def _load_signals(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _signal_score(signals: pd.DataFrame) -> Optional[float]:
    """Strength-weighted average direction, in [-1, 1]. None if no signals."""
    if signals.empty:
        return None
    direction_value = signals["direction"].map({"positive": 1, "negative": -1, "neutral": 0})
    weights = signals["strength"].clip(lower=0, upper=1)
    if weights.sum() == 0:
        return 0.0
    return float((direction_value * weights).sum() / weights.sum())


def load_company_research(
    ticker: str,
    price_prov: Optional[price_provider.PriceProvider] = None,
) -> dict:
    """Read-only aggregation of everything already known about one ticker.

    Never triggers SEC/news LLM analysis -- if signals aren't cached/saved yet,
    this simply reports fewer signals and a lower confidence, it does not go
    fetch or extract anything itself.
    """
    ticker = ticker.upper()

    sec_signals = _load_signals(config.SIGNALS_PATH)
    sec_signals = sec_signals[sec_signals["ticker"] == ticker] if not sec_signals.empty else sec_signals

    news_signals = _load_signals(config.NEWS_SIGNALS_PATH)
    news_signals = news_signals[news_signals["ticker"] == ticker] if not news_signals.empty else news_signals

    all_signals = pd.concat([sec_signals, news_signals], ignore_index=True) if not (sec_signals.empty and news_signals.empty) else pd.DataFrame()

    signals_by_type: dict[str, dict] = {}
    if not all_signals.empty:
        for signal_type, group in all_signals.groupby("signal_type"):
            signals_by_type[signal_type] = {
                "count": len(group),
                "avg_strength": float(group["strength"].mean()),
                "score": _signal_score(group),
            }

    risks = []
    catalysts = []
    if not all_signals.empty:
        risk_rows = all_signals[all_signals["signal_type"] == "risk"]
        risks = risk_rows.sort_values("strength", ascending=False)["evidence"].head(5).tolist()

        catalyst_rows = all_signals[
            (all_signals["direction"] == "positive") & (all_signals["strength"] >= 0.6)
        ]
        catalysts = catalyst_rows.sort_values("strength", ascending=False)["evidence"].head(5).tolist()

    features = {}
    try:
        prices = price_provider.get_price_history(ticker, provider=price_prov)
        features = market_features.compute_features(prices)
    except Exception:
        features = {}

    data_freshness = None
    if not all_signals.empty and "filing_date" in all_signals.columns:
        dates = pd.to_datetime(all_signals["filing_date"], errors="coerce").dropna()
        if not dates.empty:
            data_freshness = str(dates.max().date())

    return {
        "ticker": ticker,
        "signal_count": len(all_signals),
        "signals_by_type": signals_by_type,
        "research_score": _signal_score(all_signals),
        "risks": risks,
        "catalysts": catalysts,
        "market_features": features,
        "data_freshness": data_freshness,
    }
