"""Does a signal order stocks by forward return? Cross-sectional evaluation.

For each rebalance date, across the tickers that have both a signal value
and a label:
- rank IC      Spearman correlation(signal, fwd_excess) -- the headline metric
- IC           Pearson correlation
- spread       mean fwd_excess of the top quantile minus the bottom quantile
- top hit rate share of top-quantile names that beat the benchmark

Then across dates: mean, share of dates with IC > 0, and a t-stat with a
Newey-West (HAC) standard error -- monthly rebalancing with 3-month labels
means consecutive dates share 2 months of the same returns, so the naive
t-stat overstates significance roughly by sqrt(3).

`walk_forward_splits` is here for the ML stage: expanding-window train sets
with an embargo of `ceil(label_days / rebalance_every)` dates, so no training
label overlaps a test date's return window.
"""

from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

import screener

MIN_NAMES_PER_DATE = 10
DEFAULT_QUANTILES = 5
LABEL_ORDER = ["Strong", "Favorable", "Neutral", "Unfavorable", "Insufficient data"]


def per_date_metrics(
    panel: pd.DataFrame, signal: str, label: str = "fwd_excess", quantiles: int = DEFAULT_QUANTILES,
    min_names: int = MIN_NAMES_PER_DATE,
) -> pd.DataFrame:
    rows = []
    for date, group in panel.dropna(subset=[signal, label]).groupby("date"):
        if len(group) < max(min_names, quantiles * 2):
            continue
        x, y = group[signal].astype(float), group[label].astype(float)
        if x.nunique() < 2:
            continue
        buckets = pd.qcut(x.rank(method="first"), quantiles, labels=False)
        top, bottom = y[buckets == quantiles - 1], y[buckets == 0]
        rows.append({
            "date": date,
            "n": len(group),
            "rank_ic": x.corr(y, method="spearman"),
            "ic": x.corr(y),
            "spread": top.mean() - bottom.mean(),
            "top_mean": top.mean(),
            "bottom_mean": bottom.mean(),
            "top_hit_rate": (top > 0).mean(),
        })
    return pd.DataFrame(rows)


def newey_west_tstat(values: Sequence[float], lags: int) -> Optional[float]:
    x = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    n = len(x)
    if n < 3:
        return None
    d = x - x.mean()
    variance = d @ d / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * (d[lag:] @ d[:-lag]) / n
    if variance <= 0:
        return None
    return float(x.mean() / math.sqrt(variance / n))


def summarize(per_date: pd.DataFrame, overlap_lags: int) -> dict:
    if per_date.empty:
        return {"dates": 0}
    return {
        "dates": len(per_date),
        "avg_names": round(per_date["n"].mean(), 1),
        "mean_rank_ic": per_date["rank_ic"].mean(),
        "rank_ic_tstat_nw": newey_west_tstat(per_date["rank_ic"].tolist(), overlap_lags),
        "pct_dates_ic_pos": (per_date["rank_ic"] > 0).mean(),
        "mean_ic": per_date["ic"].mean(),
        "mean_spread": per_date["spread"].mean(),
        "spread_tstat_nw": newey_west_tstat(per_date["spread"].tolist(), overlap_lags),
        "top_hit_rate": per_date["top_hit_rate"].mean(),
    }


def label_buckets(panel: pd.DataFrame, label: str = "fwd_excess", score: str = "score_full") -> pd.DataFrame:
    """Mean forward excess return per app label (Strong/Favorable/...).

    The direct test of the UI: if Strong doesn't beat Unfavorable, the
    labels on the Opportunities and Quick Picks pages are noise.
    """
    data = panel.dropna(subset=[label]).copy()
    data["bucket"] = data[score].map(screener.label_for)
    grouped = data.groupby("bucket")[label].agg(["count", "mean", "median", lambda s: (s > 0).mean()])
    grouped.columns = ["count", "mean_fwd_excess", "median_fwd_excess", "share_beating_benchmark"]
    return grouped.reindex([b for b in LABEL_ORDER if b in grouped.index])


def by_year(per_date: pd.DataFrame) -> pd.DataFrame:
    if per_date.empty:
        return per_date
    frame = per_date.assign(year=pd.to_datetime(per_date["date"]).dt.year)
    return frame.groupby("year").agg(dates=("rank_ic", "size"), mean_rank_ic=("rank_ic", "mean"), mean_spread=("spread", "mean"))


def overlap_lags(label_days: int, rebalance_every: int) -> int:
    return max(0, math.ceil(label_days / rebalance_every) - 1)


def walk_forward_splits(
    dates: Sequence, min_train: int, test_size: int, embargo: int
) -> Iterator[tuple[list, list]]:
    """Expanding-window (train_dates, test_dates) pairs, in time order.

    `embargo` dates are dropped between the end of train and the start of
    test, so a training label's forward window can't reach into test.
    """
    dates = sorted(pd.Timestamp(d) for d in set(dates))
    start = min_train + embargo
    while start < len(dates):
        train = dates[: start - embargo]
        test = dates[start : start + test_size]
        if not test:
            break
        yield train, test
        start += test_size


def evaluate_signals(panel: pd.DataFrame, signals: Sequence[str], label_days: int, rebalance_every: int) -> pd.DataFrame:
    lags = overlap_lags(label_days, rebalance_every)
    rows = {s: summarize(per_date_metrics(panel, s), lags) for s in signals if s in panel.columns}
    return pd.DataFrame(rows).T
