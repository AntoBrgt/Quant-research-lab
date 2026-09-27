"""STEP 12 -- a learned ranking vs the hand-weighted one, out of sample only.

The STEP 11b backtest (point-in-time 13F universe) found no robust edge in
the hand-weighted `rank_score`. Before rewriting its weights by hand, this
asks the narrower question: can a model *learn* a better cross-sectional
ordering from the same point-in-time features -- judged only on dates it
never trained on?

Design choices, and why:
- **PIT panel only.** The current-universe panel is survivorship-biased (see
  STEP 11b); a model trained on it would learn "be a future survivor".
  `require_pit_panel` refuses anything else.
- **Explicit feature whitelist** (`FEATURES`), checked against `FORBIDDEN`:
  labels (fwd_*), the scores under comparison (score_*), the app label, and
  identifiers can never reach the feature matrix, even by a typo.
- **Cross-sectional ranks, per date**, for features and target. Only the
  ordering within a date matters to a ranking, and ranks are immune to the
  regime shifts (2020 volatility, 2022 rate moves) that would otherwise make
  raw levels mean different things in different years. NaN stays NaN
  (LightGBM routes missing values itself; fundamentals only exist from ~2022).
- **Fixed, regularized LightGBM params** (`LGBM_PARAMS`): small trees (15
  leaves), >= 100 rows per leaf, slow learning rate, row/feature bagging,
  L2. ~100 names x ~40-90 dates is a small, noisy dataset -- these are
  deliberately conservative defaults and are NOT tuned on test folds (any
  tuning would have to happen inside the training window).
- **Linear baseline** (`score_linear`): ridge on the same rank features
  (NaN -> 0.5, the cross-sectional middle). If LightGBM can't beat a linear
  model, its non-linearity isn't earning anything.
- **Walk-forward** (`evaluate.walk_forward_splits`): expanding window, and an
  embargo of overlapping-label dates + 1, so no training label's return
  window reaches into a test date's. Rows never in a test fold keep NaN
  predictions -- there is no in-sample score anywhere in the output.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from backtest import evaluate

logger = logging.getLogger(__name__)

TECHNICAL_FEATURES = [
    "return_20d", "return_60d", "return_252d",
    "relative_return_20d", "relative_return_60d", "relative_return_252d",
    "volatility_20d", "volatility_60d", "volume_ratio",
    "price_vs_ma50", "price_vs_ma200", "momentum_12_1",
]
FUNDAMENTAL_FEATURES = [
    "revenue_growth", "eps_growth", "fcf_growth",
    "net_margin", "fcf_margin", "roe", "fcf_yield", "net_debt",
]
INSTITUTIONAL_FEATURES = ["institutional_direction_score", "institution_count"]
# STEP 13: new information, not new transforms of prices.
INSIDER_FEATURES = ["insider_buyers_90d", "insider_net_buy_value_90d_mcap", "insider_cluster_buy"]  # insider_form4.py
ACTIVE_FEATURES = ["active_new_or_add", "active_weight_max"]  # concentrated managers' best ideas
# Explicit size: institution_count grows with market cap, so without this the
# model could learn "big company" through institution_count and we'd misread it.
SIZE_FEATURES = ["log_market_cap"]
STEP13_FEATURES = INSIDER_FEATURES + ACTIVE_FEATURES + SIZE_FEATURES
FEATURES = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + INSTITUTIONAL_FEATURES + STEP13_FEATURES

# Never a feature: the label, anything derived from it, the scores being
# compared against, and identifiers.
FORBIDDEN = {"fwd_return", "fwd_excess", "label", "date", "ticker", "target"}
FORBIDDEN_PREFIXES = ("fwd_", "score_")

TARGET = "fwd_excess"

LGBM_PARAMS = {
    "objective": "regression",
    "num_leaves": 15,
    "min_data_in_leaf": 100,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 0,
    "deterministic": True,
    "num_threads": 1,
    "verbose": -1,
}
NUM_TREES = 300
RIDGE_ALPHA = 1.0  # on [0,1] rank features and thousands of rows: a light, stabilizing penalty

MIN_TRAIN_DATES = 36
TEST_SIZE = 6


def check_features(features) -> list[str]:
    """Raise if any forbidden column is in the feature list."""
    bad = [f for f in features if f in FORBIDDEN or f.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise ValueError(f"Forbidden column(s) in the feature list: {bad}")
    return list(features)


check_features(FEATURES)  # at import: a bad edit to FEATURES fails loudly, not silently


def require_pit_panel(panel: pd.DataFrame) -> None:
    """The model is only ever trained on the point-in-time universe panel."""
    if "score_rank" not in panel.columns or "institutional_direction_score" not in panel.columns:
        raise ValueError("model.py needs the --universe pit panel (STEP 11b); the current-universe panel is survivorship-biased")


def with_institution_count(panel: pd.DataFrame, membership: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Panels built before STEP 12 lack `institution_count`; it is point-in-time in universe_membership.parquet."""
    if "institution_count" in panel.columns or membership is None or membership.empty:
        return panel
    counts = membership.dropna(subset=["ticker"])[["date", "ticker", "institution_count"]]
    counts = counts.assign(date=pd.to_datetime(counts["date"]))
    return panel.merge(counts, on=["date", "ticker"], how="left")


def cross_sectional_rank(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-date rank of each column, scaled to [0, 1]; NaN stays NaN.

    (rank - 1) / (n - 1) over the non-missing values of that date; a date
    with a single value gets 0.5.
    """
    grouped = frame.groupby("date")[columns]
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    return ((ranks - 1) / (counts - 1).where(counts > 1)).where(counts != 1, 0.5).where(frame[columns].notna())


def feature_matrix(panel: pd.DataFrame, features: Optional[list[str]] = None) -> pd.DataFrame:
    """Rank-transformed features, strictly from the whitelist."""
    features = check_features(features or FEATURES)
    frame = panel[["date"]].copy()
    for f in features:
        if f in panel.columns:
            frame[f] = pd.to_numeric(panel[f], errors="coerce")
        else:
            logger.warning("Feature %s not in panel: all NaN", f)
            frame[f] = np.nan
    ranked = cross_sectional_rank(frame, features)
    assert not (set(ranked.columns) & FORBIDDEN) and not any(c.startswith(FORBIDDEN_PREFIXES) for c in ranked.columns)
    return ranked[features]


def target_rank(panel: pd.DataFrame) -> pd.Series:
    return cross_sectional_rank(panel[["date", TARGET]], [TARGET])[TARGET]


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    """Closed-form ridge with an unpenalized intercept (no sklearn dependency for one solve)."""
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc = x - x_mean
    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (y - y_mean))
    return np.concatenate([[y_mean - x_mean @ coef], coef])


def _predict_ridge(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] + x @ beta[1:]


def model_splits(dates, label_days: int, rebalance_every: int, min_train: int = MIN_TRAIN_DATES, test_size: int = TEST_SIZE,
                 embargo: Optional[int] = None):
    """The walk-forward folds: embargo = overlapping-label dates + 1 (or larger, if given), so the
    last training label's return window ends before the first test date.
    """
    minimum = evaluate.overlap_lags(label_days, rebalance_every) + 1
    embargo = minimum if embargo is None else max(embargo, minimum)
    return evaluate.walk_forward_splits(dates, min_train, test_size, embargo)


def walk_forward_predict(
    panel: pd.DataFrame,
    label_days: int,
    rebalance_every: int,
    features: Optional[list[str]] = None,
    min_train: int = MIN_TRAIN_DATES,
    test_size: int = TEST_SIZE,
    num_trees: int = NUM_TREES,
    params: Optional[dict] = None,
    embargo: Optional[int] = None,
    pit_checked: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample `score_ml` / `score_linear` per (date, ticker), and feature importance.

    Returns (predictions, importance):
      predictions: date, ticker, fold, score_ml, score_linear -- NaN outside test folds
      importance:  feature, mean_gain (mean LightGBM total gain across folds)
    """
    import lightgbm as lgb

    if not pit_checked:  # a caller whose universe is point-in-time by construction (STEP 14a) says so explicitly
        require_pit_panel(panel)
    features = check_features(features or FEATURES)
    params = {**LGBM_PARAMS, **(params or {})}
    panel = panel.reset_index(drop=True)
    x = feature_matrix(panel, features)
    y = target_rank(panel)
    labelled = y.notna()

    out = panel[["date", "ticker"]].copy()
    out["fold"] = np.nan
    out["score_ml"] = np.nan
    out["score_linear"] = np.nan
    gains = []

    for fold, (train_dates, test_dates) in enumerate(
        model_splits(panel["date"].unique(), label_days, rebalance_every, min_train, test_size, embargo)
    ):
        train = panel["date"].isin(train_dates) & labelled
        test = panel["date"].isin(test_dates)
        if train.sum() < params["min_data_in_leaf"] * 2 or not test.any():
            continue
        booster = lgb.train(params, lgb.Dataset(x[train], label=y[train]), num_boost_round=num_trees)
        out.loc[test, "score_ml"] = booster.predict(x[test])
        gains.append(pd.Series(booster.feature_importance("gain"), index=features))

        beta = _fit_ridge(x[train].fillna(0.5).to_numpy(), y[train].to_numpy())
        out.loc[test, "score_linear"] = _predict_ridge(beta, x[test].fillna(0.5).to_numpy())
        out.loc[test, "fold"] = fold

    importance = (
        pd.concat(gains, axis=1).mean(axis=1).rename("mean_gain").sort_values(ascending=False)
        .rename_axis("feature").reset_index()
        if gains else pd.DataFrame(columns=["feature", "mean_gain"])
    )
    return out, importance


MIN_OOS_YEARS = 4  # the years rule needs at least this many OOS years, positive in at least this many


def ic_difference_tstat(a: pd.DataFrame, b: pd.DataFrame, lags: int) -> tuple[float, Optional[float], int]:
    """(mean, NW t-stat, dates) of the per-date rank-IC difference a - b, on dates both have.

    Beating a baseline "on average" is not enough: two noisy IC series can
    differ in mean by chance. The t-stat of the per-date difference is the
    test of *this* model being better than *that* one on the same dates.
    """
    joined = a.set_index("date")["rank_ic"].to_frame("a").join(b.set_index("date")["rank_ic"].to_frame("b"), how="inner").dropna()
    diff = joined["a"] - joined["b"]
    return (float(diff.mean()) if len(diff) else float("nan")), evaluate.newey_west_tstat(diff.tolist(), lags), len(diff)


def promotion_check(per_date: dict[str, pd.DataFrame], lags: int) -> list[str]:
    """README STEP 12 promotion rule, evaluated on the same OOS rows.

    score_ml may replace rank_score only if, per-date on the same rows, its
    rank IC beats score_rank AND momentum_12_1 with a Newey-West t-stat of the
    IC *difference* > 2 against each, and its mean rank IC is positive in at
    least MIN_OOS_YEARS OOS calendar years (which requires that many OOS years).
    """
    ml = per_date.get("score_ml")
    if ml is None or ml.empty:
        return ["- No OOS predictions: promotion rule cannot be evaluated."]
    own_t = evaluate.newey_west_tstat(ml["rank_ic"].tolist(), lags)
    lines = [f"- score_ml mean rank IC {ml['rank_ic'].mean():.4f}, own NW t = "
             + (f"{own_t:.2f}" if own_t is not None else "n/a") + " (information only; the rule tests differences)."]
    checks = []
    for baseline in ("score_rank", "momentum_12_1"):
        other = per_date.get(baseline)
        if other is None or other.empty:
            checks.append((f"beats {baseline} (NW t of IC difference > 2)", False, "baseline missing"))
            continue
        mean_diff, t, n = ic_difference_tstat(ml, other, lags)
        ok = t is not None and mean_diff > 0 and t > 2
        checks.append((f"beats {baseline} (NW t of IC difference > 2)", ok,
                       f"mean diff {mean_diff:+.4f} over {n} dates, t = " + (f"{t:.2f}" if t is not None else "n/a")))
    yearly = ml.assign(year=pd.to_datetime(ml["date"]).dt.year).groupby("year")["rank_ic"].mean()
    positive = int((yearly > 0).sum())
    checks.append((f">= {MIN_OOS_YEARS} OOS years, IC > 0 in >= {MIN_OOS_YEARS}",
                   len(yearly) >= MIN_OOS_YEARS and positive >= MIN_OOS_YEARS,
                   f"{positive} positive of {len(yearly)} OOS years"))
    lines += [f"- {'PASS' if ok else 'FAIL'} -- {name} ({detail})" for name, ok, detail in checks]
    verdict = all(ok for _, ok, _ in checks)
    lines.append(f"- **Verdict: {'score_ml qualifies for promotion' if verdict else 'score_ml does NOT qualify -- rank_score stays in screener.py'}.**")
    return lines
