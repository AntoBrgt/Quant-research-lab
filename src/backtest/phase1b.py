"""PHASE 1b: features, models and portfolios -- every loader goes through `protocol`.

Universe: the top-500 point-in-time 13F universe of STEP 15 (monthly
members with technicals, SEC-XBRL fundamentals, insider, 13F and size
features), cut at the protocol's end date. Added here:

- SUE (XBRL EPS, from the filing date) and days since the last filing;
- Lazy Prices text similarity (`lazy_prices.py`);
- industry-relative versions: feature minus its (date, SIC 2-digit) median;
- 1- and 3-month changes of each feature (same ticker, calendar months).

Target: per-date rank of the NEXT MONTH's return (next open after the
signal month-end -> next open after the following month-end). Labels come
from `protocol.load_prices`, so in development a window reaching into the
holdout has no exit price and no label.

Models (walk-forward, 36 training months, test blocks of 6, embargo 1):
LightGBM regressor and ridge (`model.walk_forward_predict`), LightGBM
lambdarank ranker with one query group per date (`walk_forward_rank`), and
rank-average ensembles.

Portfolio: long the top decile each month, equal weight, 0.15%/side on
turnover; and a volatility-targeted version scaled to 15% annual volatility
(trailing 12 months of its own returns, capped at 1.5x, cash / borrowing at
the T-bill rate) to compare with SPY at similar risk.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

import sec_xbrl
from backtest import evaluate, model, protocol, smallcap

STEP15_MEMBERS = protocol.data.BACKTEST_DIR / "phase1" / "step15" / "members.parquet"
TRAIN_MONTHS, TEST_MONTHS, EMBARGO = 36, 6, 1
COST_PER_SIDE = 0.0015
VOL_TARGET, MAX_LEVERAGE, VOL_WINDOW = 0.15, 1.5, 12
TOP_QUANTILE = 0.10
RANK_LEVELS = 10  # lambdarank relevance: per-date deciles 0..9


# ----------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------


def add_sue(panel: pd.DataFrame, cik_of: dict, end=None) -> pd.DataFrame:
    """Latest SUE filed by each date, and days since the last XBRL filing -- dev tables only."""
    sue, days = np.full(len(panel), np.nan), np.full(len(panel), np.nan)
    for ticker, idx in panel.groupby("ticker").groups.items():
        cik = cik_of.get(ticker)
        if cik is None:
            continue
        table = protocol.xbrl_table(cik, end)
        if table.empty:
            continue
        s = sec_xbrl.sue_series(table).sort_values("filed")
        filed = np.sort(pd.to_datetime(table["filed"]).unique())
        for i in idx:
            d = panel.at[i, "date"]
            known = s[s["filed"] <= d]
            if len(known):
                sue[panel.index.get_loc(i)] = known["sue"].iloc[-1]
            pos = np.searchsorted(filed, np.datetime64(d), side="right") - 1
            if pos >= 0:
                days[panel.index.get_loc(i)] = (d - pd.Timestamp(filed[pos])).days
    return panel.assign(sue=sue, days_since_filing=days)


def industry_relative(panel: pd.DataFrame, features: list[str], industry: pd.Series) -> pd.DataFrame:
    """feature - median of the same (date, industry); NaN where the industry has < 3 names that date."""
    frame = panel[["date"] + features].copy()
    frame["_ind"] = industry.values
    grouped = frame.groupby(["date", "_ind"])
    med = grouped[features].transform("median")
    count = grouped[features].transform("count")
    rel = (frame[features] - med).where(count >= 3)
    rel.columns = [f"{c}_ind" for c in features]
    return rel


def changes(panel: pd.DataFrame, features: list[str], months: int) -> pd.DataFrame:
    """feature(t) - feature(t - months) for the same ticker, exact calendar months (missing month -> NaN)."""
    key = panel[["ticker"]].assign(period=panel["date"].dt.to_period("M"))
    past = key.assign(period=key["period"] + months).join(panel[features])
    merged = key.merge(past, on=["ticker", "period"], how="left", suffixes=("", "_past"))
    out = pd.DataFrame(panel[features].to_numpy(dtype=float) - merged[features].to_numpy(dtype=float),
                       index=panel.index, columns=[f"{c}_chg{months}m" for c in features])
    return out


def next_month_returns(panel: pd.DataFrame, end=None) -> tuple[pd.Series, pd.Series]:
    """(return, exit date) from the next open after each month-end to the next open after the next month-end."""
    dates = sorted(panel["date"].unique())
    nxt = dict(zip(dates, dates[1:]))
    ret = pd.Series(np.nan, index=panel.index)
    exit_dates = pd.Series(pd.NaT, index=panel.index)
    for ticker, idx in panel.groupby("ticker").groups.items():
        try:
            px = protocol.load_prices(ticker, end)
        except Exception:
            continue
        opens = px["open"].dropna()
        if opens.empty:
            continue
        for i in idx:
            d = panel.at[i, "date"]
            if d not in nxt:
                continue
            a = opens.index.searchsorted(d, side="right")
            b = opens.index.searchsorted(nxt[d], side="right")
            if b < len(opens) and a < b:
                ret.at[i] = opens.iloc[b] / opens.iloc[a] - 1
                exit_dates.at[i] = opens.index[b]
    protocol.assert_labels_dev(exit_dates.dropna())
    return ret, exit_dates


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------


def walk_forward_rank(panel: pd.DataFrame, features: list[str], num_trees: int = model.NUM_TREES) -> pd.Series:
    """LightGBM lambdarank, one query group per date, relevance = per-date decile of the target (0..9)."""
    import lightgbm as lgb

    panel = panel.reset_index(drop=True)
    x = model.feature_matrix(panel, features)
    target = model.target_rank(panel)
    labelled = target.notna()
    relevance = np.floor(target.clip(0, 0.999999) * RANK_LEVELS)
    params = {**model.LGBM_PARAMS, "objective": "lambdarank", "metric": "ndcg", "label_gain": list(range(RANK_LEVELS))}
    params.pop("min_data_in_leaf", None)
    params["min_data_in_leaf"] = 50
    out = pd.Series(np.nan, index=panel.index)
    for train_dates, test_dates in model.model_splits(panel["date"].unique(), 21, 21, TRAIN_MONTHS, TEST_MONTHS, EMBARGO):
        train = panel["date"].isin(train_dates) & labelled
        test = panel["date"].isin(test_dates)
        if train.sum() < 500 or not test.any():
            continue
        order = panel.loc[train].sort_values("date").index
        groups = panel.loc[order].groupby("date", sort=True).size().to_numpy()
        booster = lgb.train(params, lgb.Dataset(x.loc[order], label=relevance.loc[order].astype(int), group=groups),
                            num_boost_round=num_trees)
        out.loc[test] = booster.predict(x[test])
    return out


def per_date_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return values.groupby(dates).rank(pct=True)


# ----------------------------------------------------------------------------
# Portfolio + evaluation
# ----------------------------------------------------------------------------


def portfolio(rows: pd.DataFrame, score: str, cost: float = COST_PER_SIDE) -> pd.Series:
    """Monthly net return of the top decile by `score` (equal weight, costs on turnover), indexed by signal date."""
    port = smallcap.monthly_portfolio(rows, score, rows["next_return"], top_quantile=TOP_QUANTILE, max_weight=1.0, cost=cost)
    return port["net"]


def vol_targeted(net: pd.Series, cash: pd.Series, target: float = VOL_TARGET, cap: float = MAX_LEVERAGE,
                 window: int = VOL_WINDOW) -> pd.Series:
    """Scale each month's exposure to `target` annual vol using only the PREVIOUS `window` months;
    the uninvested part earns (and leverage above 1 pays) the cash rate."""
    realized = net.rolling(window, min_periods=6).std().shift(1) * math.sqrt(12)
    exposure = (target / realized).clip(upper=cap).fillna(1.0)
    c = cash.reindex(net.index).fillna(0.0)
    return exposure * net + (1 - exposure) * c


def evaluate_signal(rows: pd.DataFrame, score: str) -> dict:
    per_date = evaluate.per_date_metrics(rows, score, label="next_return")
    s = evaluate.summarize(per_date, 0)
    return {"mean_rank_ic": s.get("mean_rank_ic", np.nan), "ic_tstat": s.get("rank_ic_tstat_nw", np.nan),
            "per_date": per_date}


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    growth = float((1 + r).prod())
    years = len(r) / 12
    return {"months": len(r), "annualized": growth ** (1 / years) - 1, "vol": float(r.std() * math.sqrt(12)),
            "sharpe": float(r.mean() / r.std() * math.sqrt(12)) if r.std() > 0 else np.nan,
            "max_drawdown": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())}
