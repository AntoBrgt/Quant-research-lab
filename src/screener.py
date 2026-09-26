"""Rank the whole institutional universe for one investment horizon.

The front page's question is "for a 3-day (or 6-month, or 10-year) horizon,
which companies in my universe currently have the strongest evidence?".
This module answers it by running the existing per-company pipeline
(`research_engine.load_extended_research`) over every ticker in the
universe and sorting by the horizon-weighted score. No new scoring logic:
the ranking is exactly the `horizon_weighted_view` the Company Research page
already shows, plus a small, visible institutional tilt.

    rank_score = horizon_weighted_score + INSTITUTIONAL_TILT * institutional_direction_score

`horizon_weighted_score` is in [-1, 1] (technical + fundamental groups,
weighted by horizon profile); `institutional_direction_score` is the
universe's mean of +1/0/-1 directions. With a 0.10 tilt, institutional
activity can break ties and nudge, but never outrank the evidence.

What the labels mean -- and don't: `Strong`/`Favorable` mean the evidence for
this horizon currently leans positive. They are not a forecast of return and
the method has not been backtested yet (see README "Known gaps").
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd

import research_engine

logger = logging.getLogger(__name__)

INSTITUTIONAL_TILT = 0.10
STRONG_THRESHOLD = 0.30
FAVORABLE_THRESHOLD = 0.10
MAX_WORKERS = 6

RANKING_COLUMNS = [
    "rank", "ticker", "company_name", "sector", "label", "rank_score", "horizon_score",
    "technical_contribution", "fundamental_contribution", "dominant_factors",
    "technical_view", "fundamental_view", "institutional_view", "institutions",
    "confidence", "current_price", "return_20d", "stop_loss", "take_profit", "risk_reward_ratio",
    "data_status", "shortlisted", "error",
]


def label_for(score: Optional[float]) -> str:
    if score is None or pd.isna(score):
        return "Insufficient data"
    if score >= STRONG_THRESHOLD:
        return "Strong"
    if score >= FAVORABLE_THRESHOLD:
        return "Favorable"
    if score <= -FAVORABLE_THRESHOLD:
        return "Unfavorable"
    return "Neutral"


def _row_for(ticker: str, horizon_days: int, universe_row: dict, mentions: pd.DataFrame, load: Callable) -> dict:
    base = {
        "ticker": ticker,
        "company_name": universe_row.get("company_name"),
        "sector": universe_row.get("sector"),
        "institutions": universe_row.get("institution_count"),
    }
    try:
        extended = load(ticker, horizon_days, institutional_mentions=mentions)
    except Exception as exc:  # one bad ticker (delisted, no data) never breaks the ranking
        logger.warning("Screener: %s failed: %s", ticker, exc)
        return base | {"label": "Insufficient data", "error": str(exc), "shortlisted": False}

    hv = extended["horizon_weighted_view"]
    views = research_engine.component_views(extended)
    confidence = research_engine.research_confidence(views, extended["research"].get("signal_count", 0))
    rr = extended["risk_reward"]
    mf = extended["research"].get("market_features", {}) or {}

    horizon_score = hv.get("score")
    inst_score = universe_row.get("institutional_direction_score") or 0.0
    rank_score = None if horizon_score is None else round(horizon_score + INSTITUTIONAL_TILT * float(inst_score), 4)
    label = label_for(rank_score)
    data_status = extended["data_freshness"]["status"]

    return base | {
        "sector": base["sector"] or extended["fundamentals"].get("sector"),
        "label": label,
        "rank_score": rank_score,
        "horizon_score": horizon_score,
        "technical_contribution": hv.get("technical_contribution"),
        "fundamental_contribution": hv.get("fundamental_contribution"),
        "dominant_factors": ", ".join(f.replace("_", " ") for f in hv.get("dominant_factors", [])),
        "technical_view": views["technical"],
        "fundamental_view": views["fundamental"],
        "institutional_view": views["institutional"],
        "confidence": confidence["level"],
        "current_price": extended.get("current_price"),
        "return_20d": mf.get("return_20d"),
        "stop_loss": rr["stop_loss"].get("level"),
        "take_profit": rr["take_profit"].get("level"),
        "risk_reward_ratio": rr.get("risk_reward_ratio"),
        "data_status": data_status,
        "shortlisted": label in ("Strong", "Favorable") and data_status == "OK" and confidence["level"] != "Insufficient evidence",
        "error": None,
    }


def rank_universe(
    universe_df: pd.DataFrame,
    horizon_days: int,
    mentions: Optional[pd.DataFrame] = None,
    load: Callable = research_engine.load_extended_research,
    prefetch: Optional[Callable[[str], object]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """One row per universe ticker, best evidence for this horizon first.

    `prefetch` (e.g. `price_provider.get_price_history`) is called once per
    ticker *sequentially* before the parallel pass -- yfinance's bulk download
    isn't safe to call from several threads at once, while the per-ticker
    fundamentals lookups that follow are.
    """
    if universe_df is None or universe_df.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    mentions = mentions if mentions is not None else pd.DataFrame()
    rows_by_ticker = {r["ticker"]: r for r in universe_df.to_dict("records")}
    tickers = list(rows_by_ticker)
    total = len(tickers)

    if prefetch is not None:
        for i, ticker in enumerate(tickers):
            try:
                prefetch(ticker)
            except Exception as exc:
                logger.warning("Screener prefetch %s failed: %s", ticker, exc)
            if progress:
                progress(i + 1, total * 2, ticker)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_row_for, t, horizon_days, rows_by_ticker[t], mentions, load): t for t in tickers}
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if progress:
                progress((total if prefetch else 0) + done, total * 2 if prefetch else total, futures[future])

    ranking = pd.DataFrame(results).reindex(columns=RANKING_COLUMNS)
    ranking["_sort"] = ranking["rank_score"].astype(float).fillna(-999)
    ranking = ranking.sort_values(["_sort", "institutions"], ascending=[False, False]).drop(columns="_sort").reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking["shortlisted"] = ranking["shortlisted"].fillna(False).astype(bool)
    return ranking


def _cache_path(horizon_days: int, universe_path) -> "Path":
    from datetime import date
    from pathlib import Path

    import config

    stamp = int(Path(universe_path).stat().st_mtime) if Path(universe_path).exists() else 0
    return config.PROCESSED_DATA_DIR / "rankings" / f"h{horizon_days}_{date.today().isoformat()}_{stamp}.parquet"


def load_cached_ranking(horizon_days: int, universe_path) -> Optional[pd.DataFrame]:
    """Today's ranking for this horizon + this exact universe build, if already computed."""
    path = _cache_path(horizon_days, universe_path)
    return pd.read_parquet(path) if path.exists() else None


def save_ranking(ranking: pd.DataFrame, horizon_days: int, universe_path) -> None:
    path = _cache_path(horizon_days, universe_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(path, index=False)
