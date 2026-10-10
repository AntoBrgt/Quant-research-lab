"""PHASE 1b addendum (2026-10-10): a verdict rule for the 11 registered variants, fixed before the holdout.

The first PHASE 1b run registered 11 of 24 planned variants and stopped; none was
promoted, so the holdout (2024-01-01 -> data end) was never opened. This addendum
closes PHASE 1b by judging exactly those 11 -- no new variant, no parameter change --
on the holdout months, once (`protocol.addendum_holdout`).

A variant PASSES only if, on holdout signal months (signal date >= HOLDOUT_START):
1. its per-date rank IC has NW t (lags = NW_LAGS) whose two-sided p-value, Holm-adjusted
   over the 11 variants, is below HOLM_ALPHA (= the p of |t| = 2 for a single test), with t > 0;
2. its mean rank IC is > 0;
3. its top-minus-bottom DECILE spread after costs (monthly, 0.15%/side on each leg's
   turnover) has a mean > 0.

Lazy Prices (the text features) PASSES only if at least one of the three
"base+sue+text" variants passes AND its holdout mean rank IC is higher than the same
model on "base+sue" (otherwise the pass could come from the non-text features alone).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest import evaluate, smallcap

ADDENDUM_DATE = "2026-10-10"
TEST_ID = "phase1b_addendum_2026-10-10"
NW_LAGS = 1
HOLM_ALPHA = float(2 * (1 - norm.cdf(2.0)))  # 0.0455: |t| = 2 for a single test
DECILE = 0.10
COST_PER_SIDE = 0.0015

# The 11 registered variants (registry.parquet, all from the 2026-09-27 run), as (model, feature set).
VARIANTS = {
    "lgbm_reg:32b5ce1048:0": ("lgbm_reg", "base"),
    "ridge:9718710769:0": ("ridge", "base"),
    "lgbm_rank:8bfa0b818b:0": ("lgbm_rank", "base"),
    "ensemble:4db711bb3f:0": ("ensemble", "base"),
    "lgbm_reg:4e59bfdeef:0": ("lgbm_reg", "base+sue"),
    "ridge:e52c1353c6:0": ("ridge", "base+sue"),
    "lgbm_rank:73997626a1:0": ("lgbm_rank", "base+sue"),
    "ensemble:bee3d697c7:0": ("ensemble", "base+sue"),
    "lgbm_reg:7bb578be22:0": ("lgbm_reg", "base+sue+text"),
    "ridge:6190916c75:0": ("ridge", "base+sue+text"),
    "lgbm_rank:900efcac6e:0": ("lgbm_rank", "base+sue+text"),
}


def two_sided_p(t: Optional[float]) -> float:
    return float(2 * (1 - norm.cdf(abs(t)))) if t is not None and np.isfinite(t) else 1.0


def holm_adjust(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def decile_spread(rows: pd.DataFrame, score: str, label: str = "next_return", cost: float = COST_PER_SIDE) -> pd.Series:
    """Monthly top-minus-bottom decile return after costs: long leg net of its costs, short leg pays its own."""
    top = smallcap.monthly_portfolio(rows, score, rows[label], top_quantile=DECILE, max_weight=1.0, cost=cost)
    flipped = rows.assign(_neg=-rows[score])
    bottom = smallcap.monthly_portfolio(flipped, "_neg", rows[label], top_quantile=DECILE, max_weight=1.0, cost=cost)
    return (top["net"] - bottom["gross"] - bottom["turnover"] * cost).rename("spread_net")


def ic_stats(rows: pd.DataFrame, score: str, label: str = "next_return") -> dict:
    per_date = evaluate.per_date_metrics(rows, score, label=label)
    if per_date.empty:
        return {"months": 0, "mean_rank_ic": float("nan"), "ic_t": None, "pct_months_ic_pos": float("nan"), "per_date": per_date}
    return {"months": len(per_date), "mean_rank_ic": float(per_date["rank_ic"].mean()),
            "ic_t": evaluate.newey_west_tstat(per_date["rank_ic"].tolist(), NW_LAGS),
            "pct_months_ic_pos": float((per_date["rank_ic"] > 0).mean()), "per_date": per_date}


def verdicts(stats: dict) -> dict:
    """stats[variant_id] = {"mean_rank_ic", "ic_t", "spread_mean"} -> per-variant checks and PASS / NO EDGE."""
    adjusted = holm_adjust({v: two_sided_p(s["ic_t"]) for v, s in stats.items()})
    out = {}
    for v, s in stats.items():
        checks = {
            "holm_t_above_2": bool(s["ic_t"] is not None and s["ic_t"] > 0 and adjusted[v] < HOLM_ALPHA),
            "mean_ic_positive": bool(np.isfinite(s["mean_rank_ic"]) and s["mean_rank_ic"] > 0),
            "spread_after_costs_positive": bool(np.isfinite(s["spread_mean"]) and s["spread_mean"] > 0),
        }
        out[v] = {"holm_p": adjusted[v], "checks": checks, "passed": all(checks.values())}
    return out


def lazy_prices_verdict(stats: dict, verdict: dict) -> dict:
    """PASS only if a base+sue+text variant passes AND beats the same model on base+sue in holdout mean IC."""
    by_spec = {VARIANTS[v]: v for v in stats}
    detail = {}
    for model in ("lgbm_reg", "ridge", "lgbm_rank"):
        text_id, base_id = by_spec.get((model, "base+sue+text")), by_spec.get((model, "base+sue"))
        if text_id is None or base_id is None:
            continue
        detail[model] = {"text_passes": verdict[text_id]["passed"],
                         "ic_text": stats[text_id]["mean_rank_ic"], "ic_without_text": stats[base_id]["mean_rank_ic"],
                         "text_adds": bool(stats[text_id]["mean_rank_ic"] > stats[base_id]["mean_rank_ic"])}
    passed = any(d["text_passes"] and d["text_adds"] for d in detail.values())
    return {"passed": passed, "verdict": "PASSED" if passed else "FAILED", "by_model": detail}


def annualized(monthly: pd.Series) -> float:
    r = monthly.dropna()
    if r.empty:
        return float("nan")
    growth = float((1 + r).prod())
    return growth ** (12 / len(r)) - 1 if growth > 0 else -1.0
