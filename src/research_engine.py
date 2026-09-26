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
import fundamentals as fundamentals_mod
import horizon as horizon_mod
import market_features
import price_provider
import risk_reward
import technicals as technicals_mod


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


# ---------------------------------------------------------------------------
# Extended, horizon-aware research (institutional universe / company page)
# ---------------------------------------------------------------------------
#
# `load_company_research` above is unchanged and keeps serving the existing
# portfolio flow (app.py, strategy.py, recommendations.py) exactly as before.
# The functions below are additive: they compose it with fundamentals,
# institutional context, horizon-aware scoring, and the risk/reward framework
# into the single object the company-research page needs, without touching
# the tested behavior above.

# Days old before a data source is considered stale for that horizon bucket.
# Per brief section 19: stale *price* data matters far more for a 1-day
# horizon than a 10-year one, so the threshold is horizon-dependent, not one
# global constant.
_STALENESS_THRESHOLD_DAYS = {
    "very_short_term": {"price": 1, "fundamentals": 200, "institutional": 400},
    "short_term": {"price": 5, "fundamentals": 200, "institutional": 400},
    "medium_term": {"price": 20, "fundamentals": 200, "institutional": 400},
    "long_term": {"price": 90, "fundamentals": 400, "institutional": 800},
}


def _days_since(iso_date: Optional[str]) -> Optional[int]:
    if not iso_date:
        return None
    parsed = pd.to_datetime(iso_date, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return (pd.Timestamp.now(tz="UTC") - parsed).days


def assess_data_freshness(
    horizon_days: int,
    last_price_update: Optional[str],
    last_fundamental_update: Optional[str],
    last_institutional_report_update: Optional[str],
) -> dict:
    """Per-source freshness + one overall STALE/OK flag for the chosen horizon.

    Which source's staleness actually matters depends on the horizon: a
    5-day-old price is disqualifying for a 1-day trade and irrelevant for a
    10-year thesis, while a 300-day-old institutional report is fine for a
    day trade and a real gap for a long-term thesis.
    """
    category = horizon_mod.horizon_category(horizon_days)
    thresholds = _STALENESS_THRESHOLD_DAYS[category]

    price_age = _days_since(last_price_update)
    fundamentals_age = _days_since(last_fundamental_update)
    institutional_age = _days_since(last_institutional_report_update)

    price_stale = price_age is not None and price_age > thresholds["price"]
    fundamentals_stale = fundamentals_age is not None and fundamentals_age > thresholds["fundamentals"]
    institutional_stale = institutional_age is not None and institutional_age > thresholds["institutional"]

    if category in ("very_short_term", "short_term"):
        overall_stale = price_stale
    else:
        overall_stale = fundamentals_stale or institutional_stale

    return {
        "horizon_category": category,
        "last_price_update": last_price_update,
        "last_fundamental_update": last_fundamental_update,
        "last_institutional_report_update": last_institutional_report_update,
        "price_stale": price_stale,
        "fundamentals_stale": fundamentals_stale,
        "institutional_stale": institutional_stale,
        "status": "STALE" if overall_stale else "OK",
    }


def load_extended_research(
    ticker: str,
    horizon_days: int,
    institutional_mentions: Optional[pd.DataFrame] = None,
    price_prov: Optional[price_provider.PriceProvider] = None,
    fundamentals_prov: Optional[fundamentals_mod.FundamentalsProvider] = None,
    benchmark_prov: Optional[price_provider.PriceProvider] = None,
) -> dict:
    """Everything the company-research page needs for one ticker + horizon.

    Composes (never recomputes) `load_company_research`, `fundamentals.py`,
    `technicals.py`, `institutional_research.universe.why_in_universe`,
    `horizon.py`, and `risk_reward.py`. Read-only and cache-first throughout --
    this triggers no new SEC/news LLM extraction (same contract as
    `load_company_research`) and no new institutional-report extraction; a
    ticker with nothing cached yet simply reports thinner evidence, not an
    error.
    """
    ticker = ticker.upper()
    research = load_company_research(ticker, price_prov=price_prov)

    fundamentals = fundamentals_mod.compute_fundamentals(ticker, provider=fundamentals_prov)
    technicals_result = technicals_mod.compute_technicals(ticker, price_prov=price_prov, benchmark_prov=benchmark_prov)

    # technicals_result is an additive superset of market_features.compute_features()'s
    # keys -- every name they share (atr_14d, moving_average_50d/200d,
    # price_vs_ma50/200, rsi_14d, volume_ratio, support_60d/resistance_60d, ...)
    # comes from calling the exact same market_features.py functions with the
    # exact same windows, so merging it in enriches what horizon.py/
    # risk_reward.py/component_views already read (return_120d, return_252d,
    # SMA100, volatility_20d/60d, relative strength, trend states, ...)
    # without changing the value or presence of anything they already relied on.
    research["market_features"] = {**research.get("market_features", {}), **technicals_result}

    current_price = technicals_result.get("current_price")
    last_price_update = technicals_result.get("as_of")

    # compute_fundamentals already stamps its own data_freshness from the same
    # cache-file mtime `fundamentals_mod.last_updated` would recompute --
    # reuse it rather than calling the same underlying check twice.
    last_fundamental_update = fundamentals.get("data_freshness")

    if institutional_mentions is None or institutional_mentions.empty:
        why_in_universe = pd.DataFrame()
        last_institutional_report_update = None
    else:
        from institutional_research import universe as universe_mod

        why_in_universe = universe_mod.why_in_universe(ticker, institutional_mentions)
        dates = pd.to_datetime(why_in_universe.get("publication_date"), errors="coerce").dropna() if not why_in_universe.empty else pd.Series(dtype="datetime64[ns]")
        last_institutional_report_update = str(dates.max().date()) if not dates.empty else None

    horizon_fit = horizon_mod.score_horizon_fit(research, fundamentals, horizon_days)
    # Additive, explainable companion to horizon_fit above: 5 transparent
    # profiles (VERY_SHORT..VERY_LONG) weighting 10 named component groups,
    # rather than ~15 individually-weighted raw fields. Interprets/weights
    # the already-computed technicals/fundamentals -- it does not recompute
    # or modify either.
    horizon_weighted_view = horizon_mod.compute_horizon_weighted_view(research.get("market_features", {}), fundamentals, horizon_days)
    risk = risk_reward.compute_risk_reward(current_price, research.get("market_features", {}), fundamentals, horizon_days)
    freshness = assess_data_freshness(horizon_days, last_price_update, last_fundamental_update, last_institutional_report_update)

    return {
        "ticker": ticker,
        "current_price": current_price,
        "research": research,
        "fundamentals": fundamentals,
        "institutional_context": why_in_universe,
        "horizon_fit": horizon_fit,
        "horizon_weighted_view": horizon_weighted_view,
        "risk_reward": risk,
        "data_freshness": freshness,
    }


# Component score -> label thresholds, shared by every component below so
# "POSITIVE"/"NEUTRAL"/"NEGATIVE" means the same magnitude everywhere.
_VIEW_LABEL_THRESHOLD = 0.15
# The technical/fundamental factors summarized into one label each -- a
# deliberately small, fixed set (not every available feature) so the label is
# readable as "what does this component think", not a re-derivation of the
# full horizon-weighted score.
_TECHNICAL_FACTORS = ("price_vs_ma50", "price_vs_ma200", "return_60d")
_FUNDAMENTAL_FACTORS = ("revenue_growth", "net_margin", "fcf_margin")


def _label_from_score(score: Optional[float]) -> str:
    if score is None:
        return "INSUFFICIENT_EVIDENCE"
    if score > _VIEW_LABEL_THRESHOLD:
        return "POSITIVE"
    if score < -_VIEW_LABEL_THRESHOLD:
        return "NEGATIVE"
    return "NEUTRAL"


def component_views(extended: dict) -> dict:
    """One POSITIVE/NEUTRAL/NEGATIVE/INSUFFICIENT_EVIDENCE label per component.

    Section 17's "do not hide the individual components behind one score":
    this computes the three labels the company-research page shows side by
    side (fundamental, technical, institutional) from `load_extended_research`'s
    output. Each is independent and can disagree -- that disagreement is
    itself useful information, not something to be resolved into one number
    here. `horizon_fit`'s combined score (already computed, horizon-weighted)
    is reported alongside these, not derived from them.
    """
    market_features = extended.get("research", {}).get("market_features", {}) or {}
    technical_values = [
        horizon_mod.normalize_market_feature(name, market_features[name])
        for name in _TECHNICAL_FACTORS
        if market_features.get(name) is not None
    ]
    technical_score = sum(technical_values) / len(technical_values) if technical_values else None

    fundamentals = extended.get("fundamentals", {}) or {}
    fundamental_values = [
        horizon_mod.normalize_fundamental(path, horizon_mod.get_nested(fundamentals, path))
        for path in _FUNDAMENTAL_FACTORS
        if horizon_mod.get_nested(fundamentals, path) is not None
    ]
    fundamental_score = sum(fundamental_values) / len(fundamental_values) if fundamental_values else None

    institutional_context = extended.get("institutional_context")
    if institutional_context is None or institutional_context.empty:
        institutional_label = "INSUFFICIENT_EVIDENCE"
    else:
        # Reuse the same aggregation universe.py already computes for this
        # ticker's mentions rather than re-deriving it here.
        from institutional_research.universe import aggregate_direction

        _, institutional_label = aggregate_direction(institutional_context["view_direction"])

    return {
        "fundamental": _label_from_score(fundamental_score),
        "technical": _label_from_score(technical_score),
        "institutional": institutional_label,
        "horizon_fit_score": extended.get("horizon_fit", {}).get("score"),
    }


def research_confidence(views: dict, signal_count: int) -> dict:
    """Confidence = evidence quality / component agreement, NOT probability of
    return (brief section 15). Returns {"level": High|Medium|Low|"Insufficient
    evidence", "drivers": [...]} so the reasoning is visible, not just the label.
    """
    labels = [views.get("fundamental"), views.get("technical"), views.get("institutional")]
    available = [l for l in labels if l != "INSUFFICIENT_EVIDENCE"]
    drivers = [f"{len(available)}/3 components have usable evidence (fundamental/technical/institutional)"]

    if not available:
        return {"level": "Insufficient evidence", "drivers": drivers}

    non_neutral = [l for l in available if l in ("POSITIVE", "NEGATIVE")]
    directions = set(non_neutral)
    agrees = len(directions) <= 1  # empty (all neutral) or a single shared direction

    drivers.append(f"underlying signal count: {signal_count}")
    drivers.append("components agree on direction" if agrees else "components disagree on direction (POSITIVE vs NEGATIVE)")

    if len(available) >= 2 and agrees and signal_count >= 5:
        level = "High"
    elif len(available) >= 2 and not agrees:
        level = "Medium"
        drivers.append("mixed evidence lowers confidence even though multiple components have data")
    elif len(available) == 1:
        level = "Low"
        drivers.append("only one component has usable evidence")
    else:
        level = "Medium"

    return {"level": level, "drivers": drivers}
