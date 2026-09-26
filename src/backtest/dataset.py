"""Build the (date, ticker) research panel.

One row per rebalance date x ticker with:
- point-in-time technical + fundamental features (`pit_features.py`)
- baseline scores, computed by the production `horizon.compute_horizon_weighted_view`:
    score_full       technicals + fundamentals (what the app ranks by, minus the 13F tilt)
    score_technical  technicals only -- available over the whole price history,
                     whereas yfinance fundamentals only cover the last ~4 fiscal years
- reference signals: momentum_12_1, low_volatility (= -volatility_60d)
- labels: fwd_return, fwd_excess (vs benchmark), from `labels.py`

The 13F institutional tilt (0.10 x direction) is not in the baseline: only
the latest two 13F quarters are ingested today, so it has no history to test.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

import pandas as pd

import horizon
import screener
from backtest import labels, pit_features

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_DAYS = 91  # "3 months" preset -> MEDIUM profile
DEFAULT_LABEL_DAYS = 63  # ~3 months of trading days
DEFAULT_REBALANCE_EVERY = 21  # ~monthly
WARMUP_DAYS = 252  # the 252-day return / 200-day MA need a year of history


def rebalance_dates(calendar: pd.DatetimeIndex, every: int, warmup: int = WARMUP_DAYS, start=None, end=None) -> list[pd.Timestamp]:
    dates = list(calendar.sort_values()[warmup::every])
    if start is not None:
        dates = [d for d in dates if d >= pd.Timestamp(start)]
    if end is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    return dates


def _score(technicals: dict, fundamentals: Optional[dict], horizon_days: int) -> Optional[float]:
    return horizon.compute_horizon_weighted_view(technicals or None, fundamentals, horizon_days)["score"]


def build_panel(
    tickers: Iterable[str],
    price_loader: Callable[[str], pd.DataFrame],
    fundamentals_loader: Optional[Callable[[str], dict]],
    benchmark: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    label_days: int = DEFAULT_LABEL_DAYS,
    rebalance_every: int = DEFAULT_REBALANCE_EVERY,
    fundamental_lag_days: int = pit_features.DEFAULT_FUNDAMENTAL_LAG_DAYS,
    start=None,
    end=None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    benchmark = benchmark.sort_index()
    calendar = benchmark.index
    dates = rebalance_dates(calendar, rebalance_every, start=start, end=end)
    bench_closes = benchmark["adj_close"]
    tickers = list(dict.fromkeys(t.upper() for t in tickers))

    rows: list[dict] = []
    for i, ticker in enumerate(tickers, 1):
        if progress:
            progress(i, len(tickers), ticker)
        try:
            prices = price_loader(ticker).sort_index()
        except Exception as exc:  # one bad ticker never stops the run
            logger.warning("Skipping %s: price load failed (%s)", ticker, exc)
            continue
        if prices.empty:
            continue
        raw_fund = {}
        if fundamentals_loader is not None:
            try:
                raw_fund = fundamentals_loader(ticker) or {}
            except Exception as exc:
                logger.warning("No fundamentals for %s (%s)", ticker, exc)
        stock_closes = labels.aligned_closes(prices, calendar)
        price_now = float(prices["adj_close"].iloc[-1])

        for date in dates:
            tech = pit_features.pit_technicals(prices, date, benchmark)
            if not tech:
                continue
            fund = pit_features.pit_fundamentals(
                raw_fund, date, price_at_as_of=tech.get("adj_close"), price_now=price_now, lag_days=fundamental_lag_days
            ) if raw_fund else {}
            has_fund = any(fund.get(f) is not None for fs in horizon.GROUP_FIELDS.values() for f in fs if f in fund)
            fwd_ret, fwd_excess = labels.forward_returns(stock_closes, bench_closes, date, label_days)
            score_full = _score(tech, fund if has_fund else None, horizon_days)
            vol60 = tech.get("volatility_60d")
            rows.append({
                "date": date,
                "ticker": ticker,
                **{k: v for k, v in tech.items() if k != "adj_close"},
                **{k: v for k, v in fund.items() if k not in ("market_cap",)},
                "has_fundamentals": has_fund,
                "score_full": score_full,
                "score_technical": _score(tech, None, horizon_days),
                "label": screener.label_for(score_full),
                "low_volatility": -vol60 if vol60 is not None else None,
                "fwd_return": fwd_ret,
                "fwd_excess": fwd_excess,
            })

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel
