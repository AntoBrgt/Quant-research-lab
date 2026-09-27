"""Meta-labeling: can a model tell which picks will hit their target first?

The primary signal (the app's picks) decides WHAT to buy; this model only
decides whether to TAKE a given trade -- P(target before stop/time), from the
same point-in-time features as model.py plus the trade's own geometry. A good
filter raises expectancy and cuts drawdown even if it can't rank stocks.

- Features: `model.FEATURES` as per-date ranks (same whitelist, same
  forbidden-column checks) + geometry, raw: stop distance %, target distance
  %, ATR %. Days to next earnings is NOT included: yfinance only exposes
  recent/upcoming earnings dates, not a point-in-time history, and a
  backfilled date would leak.
- Trained on every PIT member's barrier outcome (not only past picks: ~5
  picks a week is too few samples), labelled y = 1 if the target came first.
- Walk-forward over signal dates with `evaluate.walk_forward_splits`; embargo
  = max hold in rebalance units + 1, so no training trade is still open on
  the first test date. Class imbalance via scale_pos_weight (neg/pos on train).
- **Threshold chosen on training data only**: inside each training window, an
  inner split (earlier 75% fit, embargo, later 25% validate) picks the
  probability cut-off that maximizes mean R on the validation trades (with a
  minimum number of trades); the test fold never influences it.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from backtest import evaluate, model

GEOMETRY_FEATURES = ["stop_dist_pct", "target_dist_pct", "atr_pct"]
CLASSIFIER_PARAMS = {**model.LGBM_PARAMS, "objective": "binary"}
NUM_TREES = 300
MIN_TRAIN_DATES = 104  # two years of weekly signal dates
TEST_SIZE = 26  # half a year
INNER_VALIDATION_SHARE = 0.25
THRESHOLD_QUANTILES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)  # 0.0 = take everything
MIN_THRESHOLD_TRADES = 50


def embargo_dates(max_hold: int, rebalance_every: int) -> int:
    return math.ceil(max_hold / rebalance_every) + 1


def design_matrix(candidates: pd.DataFrame) -> pd.DataFrame:
    x = model.feature_matrix(candidates)
    for col in GEOMETRY_FEATURES:
        x[col] = pd.to_numeric(candidates[col], errors="coerce")
    return x


def _fit(x: pd.DataFrame, y: pd.Series, num_trees: int):
    import lightgbm as lgb

    pos = float(y.sum())
    params = {**CLASSIFIER_PARAMS, "scale_pos_weight": (len(y) - pos) / pos if pos > 0 else 1.0}
    return lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=num_trees)


def _choose_threshold(x, y, r, dates, embargo, num_trees) -> float:
    """Probability cut-off maximizing mean R on an inner validation slice of the TRAINING dates."""
    unique = sorted(dates.unique())
    cut = int(len(unique) * (1 - INNER_VALIDATION_SHARE))
    fit_dates, val_dates = set(unique[: max(cut - embargo, 1)]), set(unique[cut:])
    fit, val = dates.isin(fit_dates), dates.isin(val_dates)
    if fit.sum() < CLASSIFIER_PARAMS["min_data_in_leaf"] * 2 or y[fit].nunique() < 2 or val.sum() < MIN_THRESHOLD_TRADES:
        return 0.0
    p = _fit(x[fit], y[fit], num_trees).predict(x[val])
    best, best_r = 0.0, r[val].mean()
    for q in THRESHOLD_QUANTILES[1:]:
        thr = float(np.quantile(p, q))
        take = p >= thr
        if take.sum() >= MIN_THRESHOLD_TRADES and r[val][take].mean() > best_r:
            best, best_r = thr, r[val][take].mean()
    return best


def walk_forward_probabilities(
    candidates: pd.DataFrame, result: pd.DataFrame, max_hold: int, rebalance_every: int,
    min_train: int = MIN_TRAIN_DATES, test_size: int = TEST_SIZE, num_trees: int = NUM_TREES,
) -> pd.DataFrame:
    """OOS P(target first) per candidate + the fold's train-chosen threshold (NaN outside test folds)."""
    # Training uses trades that were actually entered; predictions cover EVERY candidate on a
    # test date -- whether the entry gaps through a barrier is next-day information and must
    # not decide which names get a probability (and so which pick moves up the list).
    x_all = design_matrix(candidates)
    valid = result["outcome"] != "skipped"
    x, y, r = x_all[valid], result.loc[valid, "y"].astype(int), result.loc[valid, "r_multiple"]
    dates, all_dates = candidates.loc[valid, "date"], candidates["date"]
    exit_pos = result.loc[valid, "exit_pos"]
    embargo = embargo_dates(max_hold, rebalance_every)

    out = pd.DataFrame({"p_target": np.nan, "threshold": np.nan, "fold": np.nan}, index=candidates.index)
    for fold, (train_dates, test_dates) in enumerate(evaluate.walk_forward_splits(all_dates.unique(), min_train, test_size, embargo)):
        train, test = dates.isin(train_dates), all_dates.isin(test_dates)
        if not test.any():
            continue
        # Belt and braces: every training trade must have exited before the first test signal.
        train &= exit_pos < int(candidates.loc[test, "signal_pos"].min())
        if train.sum() < CLASSIFIER_PARAMS["min_data_in_leaf"] * 2 or y[train].nunique() < 2:
            continue
        threshold = _choose_threshold(x[train], y[train], r[train], dates[train], embargo, num_trees)
        booster = _fit(x[train], y[train], num_trees)
        out.loc[test, "p_target"] = booster.predict(x_all[test])
        out.loc[test, "threshold"] = threshold
        out.loc[test, "fold"] = fold
    return out


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney), ties averaged."""
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def calibration_table(y: pd.Series, p: pd.Series, bins: int = 10) -> pd.DataFrame:
    decile = pd.qcut(p.rank(method="first"), bins, labels=False) + 1
    table = pd.DataFrame({"y": y, "p": p, "decile": decile}).groupby("decile").agg(
        trades=("y", "size"), mean_predicted=("p", "mean"), actual_hit_rate=("y", "mean"))
    table.index.name = "decile (10 = highest P)"
    return table
