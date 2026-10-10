"""Research protocol (PHASE 1b): iterate freely, without fooling ourselves.

Three safeguards, each aimed at a specific way backtests lie:

1. **Holdout** (`HOLDOUT_START` -> today). Every development loader here
   (prices, XBRL, filing text, the monthly member panel) either stops before
   the holdout or raises `HoldoutViolation` when asked for a date inside it.
   Labels are computed from dev prices only, so a forward window crossing into
   the holdout has no price and no label -- it can't leak by construction.
   Only `final_test(strategy_id, runner)` opens the holdout: once per
   promoted strategy, logged in `holdout_log.parquet`, never twice.

2. **Registry** (`registry.parquet`, append-only). Every variant ever run --
   model, feature set, parameters -- is logged with its dev metrics and its
   monthly dev return series. Nothing is ever removed: the number of things
   tried is part of the result.

3. **Selection-bias statistics over ALL registered variants**:
   - Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): the probability
     that the best Sharpe is above what the maximum of N unskilled trials
     would reach by luck, adjusted for skew / fat tails / sample length.
   - Probability of backtest overfitting via CSCV (Bailey, Borwein, Lopez de
     Prado & Zhu, 2017): split the dev months into S blocks; for every half/half
     split, pick the best variant in-sample and see where it ranks
     out-of-sample. PBO = share of splits where it lands in the bottom half.

Promotion to the holdout needs DSR > 0.95 and PBO < 0.2 (`PROMOTION_*`).
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest import data

HOLDOUT_START = pd.Timestamp("2024-01-01")
DEV_END = HOLDOUT_START - pd.Timedelta(days=1)
PHASE_DIR = data.BACKTEST_DIR / "phase1b"
REGISTRY_PATH = data.BACKTEST_DIR / "registry.parquet"
RETURNS_DIR = data.BACKTEST_DIR / "registry_returns"
PROMOTIONS_PATH = data.BACKTEST_DIR / "promotions.parquet"
HOLDOUT_LOG_PATH = data.BACKTEST_DIR / "holdout_log.parquet"
ADDENDUM_LOG_PATH = data.BACKTEST_DIR / "addendum_holdout_log.parquet"
PROMOTION_MIN_DSR = 0.95
PROMOTION_MAX_PBO = 0.20
CSCV_BLOCKS = 16
EULER_GAMMA = 0.5772156649

_holdout_open = False


class HoldoutViolation(RuntimeError):
    """A development step asked for data inside the holdout period."""


# ----------------------------------------------------------------------------
# Holdout guard
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def _holdout_access():
    global _holdout_open
    _holdout_open = True
    try:
        yield
    finally:
        _holdout_open = False


def holdout_is_open() -> bool:
    return _holdout_open


def assert_dev(dates, what: str = "date") -> None:
    """Raise if any date is in the holdout (unless inside final_test)."""
    if _holdout_open:
        return
    stamps = pd.to_datetime(pd.Series(list(dates) if not isinstance(dates, (pd.Series, pd.Index)) else dates)).dropna()
    if len(stamps) and stamps.max() >= HOLDOUT_START:
        raise HoldoutViolation(f"{what} {stamps.max():%Y-%m-%d} is in the holdout (>= {HOLDOUT_START:%Y-%m-%d}); "
                               "only protocol.final_test may read it")


def end_date(end=None) -> pd.Timestamp:
    """The last date a loader may return: DEV_END by default, the requested end if it's allowed."""
    if end is None:
        return pd.Timestamp.today().normalize() if _holdout_open else DEV_END
    end = pd.Timestamp(end)
    assert_dev([end], "requested end")
    return end


def load_prices(ticker: str, end=None) -> pd.DataFrame:
    """PHASE 1 price history, truncated at `end` (dev: before the holdout)."""
    prices = data.load_ext_prices(ticker)
    return prices[prices.index <= end_date(end)]


def xbrl_table(cik: int, end=None) -> pd.DataFrame:
    import sec_xbrl

    table = sec_xbrl.company_table(cik)
    return table[pd.to_datetime(table["filed"]) <= end_date(end)]


def filing_sections(cik: int, end=None) -> pd.DataFrame:
    import edgar_sections

    frame = edgar_sections.load_sections(cik)
    return frame[pd.to_datetime(frame["filing_date"]) <= end_date(end)]


def monthly_members(path, end=None) -> pd.DataFrame:
    """A monthly point-in-time panel (e.g. STEP 15 members) cut at `end`."""
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[frame["date"] <= end_date(end)].reset_index(drop=True)


def assert_labels_dev(label_end_dates) -> None:
    """Every label window must END before the holdout (entry AND exit are dev data)."""
    assert_dev(label_end_dates, "label window end")


# ----------------------------------------------------------------------------
# Append-only registry
# ----------------------------------------------------------------------------

REGISTRY_COLUMNS = ["variant_id", "family", "description", "params", "dev_start", "dev_end", "months",
                    "mean_rank_ic", "ic_tstat", "sharpe", "annualized", "timestamp"]


def _variant_id(family: str, params: dict) -> str:
    digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{family}:{digest}"


def registry() -> pd.DataFrame:
    return pd.read_parquet(REGISTRY_PATH) if REGISTRY_PATH.exists() else pd.DataFrame(columns=REGISTRY_COLUMNS)


def register(family: str, description: str, params: dict, returns: pd.Series, mean_rank_ic: float = np.nan,
             ic_tstat: float = np.nan) -> str:
    """Log one run. Append-only: a re-run of the same params gets a new row (runs are counted, not deduplicated).

    `returns` = the variant's monthly dev-period net returns (the input to DSR / PBO).
    """
    returns = returns.dropna().sort_index()
    assert_dev(returns.index, "registered return")
    base = _variant_id(family, params)
    existing = registry()
    run = int((existing["variant_id"].str.startswith(base)).sum()) if len(existing) else 0
    variant_id = f"{base}:{run}"
    std = returns.std()
    row = {
        "variant_id": variant_id, "family": family, "description": description,
        "params": json.dumps(params, sort_keys=True, default=str),
        "dev_start": returns.index.min(), "dev_end": returns.index.max(), "months": len(returns),
        "mean_rank_ic": mean_rank_ic, "ic_tstat": ic_tstat,
        "sharpe": float(returns.mean() / std * math.sqrt(12)) if std and std > 0 else np.nan,
        "annualized": float((1 + returns).prod() ** (12 / len(returns)) - 1) if len(returns) else np.nan,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RETURNS_DIR.mkdir(parents=True, exist_ok=True)
    returns.rename("ret").to_frame().to_parquet(RETURNS_DIR / f"{variant_id.replace(':', '_')}.parquet")
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_parquet(REGISTRY_PATH, index=False)
    return variant_id


def returns_matrix(variant_ids: Optional[list[str]] = None) -> pd.DataFrame:
    """Monthly dev returns of registered variants, one column each, on their common dates."""
    reg = registry()
    ids = variant_ids or reg["variant_id"].tolist()
    cols = {v: pd.read_parquet(RETURNS_DIR / f"{v.replace(':', '_')}.parquet")["ret"] for v in ids}
    return pd.DataFrame(cols).dropna()


# ----------------------------------------------------------------------------
# Deflated Sharpe ratio and PBO
# ----------------------------------------------------------------------------


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """E[max of N unskilled Sharpe ratios] (per period), Bailey & Lopez de Prado (2014)."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    return math.sqrt(sharpe_variance) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                         + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns: pd.Series, trial_sharpes, n_trials: Optional[int] = None) -> dict:
    """Probability that the true (per-period) Sharpe exceeds the luck benchmark of `n_trials` trials.

    `trial_sharpes`: per-period Sharpe of every registered variant (their
    variance sets how high luck alone can reach). Skew and kurtosis of the
    candidate's returns widen the Sharpe's standard error.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    t = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0  # pandas gives excess kurtosis
    sharpes = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    n = n_trials or len(sharpes)
    sr0 = expected_max_sharpe(n, float(sharpes.var(ddof=1)) if len(sharpes) > 1 else 0.0)
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(t - 1) / denom))
    return {"dsr": dsr, "sharpe_per_period": sr, "sharpe_annual": sr * math.sqrt(12), "benchmark_sharpe_per_period": sr0,
            "n_trials": n, "months": t, "skew": skew, "kurtosis": kurt}


def pbo_cscv(matrix: pd.DataFrame, blocks: int = CSCV_BLOCKS) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    `matrix`: rows = months, columns = variants (monthly returns). For each way
    of choosing half of `blocks` contiguous blocks as in-sample: take the
    in-sample-best variant, find its relative rank out-of-sample
    (omega in (0,1)), logit = ln(omega / (1 - omega)). PBO = P(logit <= 0).
    """
    m = matrix.dropna().to_numpy(dtype=float)
    t, n = m.shape
    if n < 2 or t < blocks:
        return {"pbo": np.nan, "combinations": 0, "variants": n, "months": t}
    t_use = t - t % blocks
    m = m[t - t_use:]
    parts = np.array_split(np.arange(t_use), blocks)
    logits, oos_best = [], []
    for combo in itertools.combinations(range(blocks), blocks // 2):
        ins = np.concatenate([parts[i] for i in combo])
        oos = np.concatenate([parts[i] for i in range(blocks) if i not in combo])
        def sharpe(rows):
            x = m[rows]
            sd = x.std(axis=0, ddof=1)
            return np.where(sd > 0, x.mean(axis=0) / np.where(sd > 0, sd, 1), -np.inf)
        s_in, s_out = sharpe(ins), sharpe(oos)
        best = int(np.argmax(s_in))
        rank = (s_out < s_out[best]).sum() + 0.5 * ((s_out == s_out[best]).sum() - 1) + 1  # 1..n
        omega = rank / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
        oos_best.append(s_out[best])
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "combinations": len(logits), "variants": n, "months": t_use,
            "median_logit": float(np.median(logits)), "p_oos_best_negative": float((np.asarray(oos_best) < 0).mean())}


def selection_stats(candidate_id: str) -> dict:
    """DSR of `candidate_id` against ALL registered variants, and PBO over all of them."""
    reg = registry()
    matrix = returns_matrix()
    per_period = [pd.read_parquet(RETURNS_DIR / f"{v.replace(':', '_')}.parquet")["ret"] for v in reg["variant_id"]]
    trial_sharpes = [r.mean() / r.std(ddof=1) for r in per_period if len(r) > 2 and r.std(ddof=1) > 0]
    candidate = pd.read_parquet(RETURNS_DIR / f"{candidate_id.replace(':', '_')}.parquet")["ret"]
    return {"candidate": candidate_id, **deflated_sharpe(candidate, trial_sharpes, n_trials=len(reg)),
            **{f"pbo_{k}" if k != "pbo" else k: v for k, v in pbo_cscv(matrix).items()}}


# ----------------------------------------------------------------------------
# Promotion and the one-shot final test
# ----------------------------------------------------------------------------


def promote(candidate_id: str) -> dict:
    """Record the promotion decision for `candidate_id` (append-only). Passing requires DSR > 0.95 and PBO < 0.2."""
    stats = selection_stats(candidate_id)
    stats["promoted"] = bool(stats["dsr"] > PROMOTION_MIN_DSR and np.isfinite(stats["pbo"]) and stats["pbo"] < PROMOTION_MAX_PBO)
    stats["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = pd.read_parquet(PROMOTIONS_PATH) if PROMOTIONS_PATH.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([stats])], ignore_index=True).to_parquet(PROMOTIONS_PATH, index=False)
    return stats


def holdout_log() -> pd.DataFrame:
    return pd.read_parquet(HOLDOUT_LOG_PATH) if HOLDOUT_LOG_PATH.exists() else pd.DataFrame(columns=["strategy_id"])


def final_test(strategy_id: str, runner: Callable[[], dict]) -> dict:
    """Run `runner` with the holdout open -- once per promoted strategy, logged either way.

    Refuses if the strategy was never promoted, or was already tested (a second
    look at the holdout would turn it into more development data).
    """
    promotions = pd.read_parquet(PROMOTIONS_PATH) if PROMOTIONS_PATH.exists() else pd.DataFrame(columns=["candidate", "promoted"])
    ok = promotions[(promotions["candidate"] == strategy_id) & promotions["promoted"].astype(bool)]
    if ok.empty:
        raise HoldoutViolation(f"{strategy_id} was not promoted (needs DSR > {PROMOTION_MIN_DSR} and PBO < {PROMOTION_MAX_PBO})")
    log = holdout_log()
    if (log["strategy_id"] == strategy_id).any():
        raise HoldoutViolation(f"{strategy_id} already had its one holdout test")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _holdout_access():
        result = runner()
    entry = {"strategy_id": strategy_id, "started": started,
             "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "summary": json.dumps(result.get("summary", {}), default=str)}
    pd.concat([log, pd.DataFrame([entry])], ignore_index=True).to_parquet(HOLDOUT_LOG_PATH, index=False)
    return result


def addendum_holdout(test_id: str, runner: Callable[[], dict], force: bool = False) -> dict:
    """The 2026-10-10 addendum's one holdout look (README, PHASE 1b addendum).

    The 11 registered variants were never promoted, so `final_test` cannot run them.
    The addendum fixed a verdict rule for all 11 at once BEFORE the holdout was
    opened; this runs it once. A second call is refused, and every call is logged
    BEFORE it runs, so a crashed run still counts as a look. `force` re-runs anyway
    (e.g. after a crash); the log and the report then say so.
    """
    log = pd.read_parquet(ADDENDUM_LOG_PATH) if ADDENDUM_LOG_PATH.exists() else pd.DataFrame(columns=["test_id"])
    previous = int((log["test_id"] == test_id).sum())
    if previous and not force:
        raise HoldoutViolation(f"{test_id} already had its one holdout look")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = {"test_id": test_id, "started": started, "finished": None, "summary": None, "run": previous + 1,
             "forced": bool(previous)}
    ADDENDUM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([log, pd.DataFrame([entry])], ignore_index=True).to_parquet(ADDENDUM_LOG_PATH, index=False)  # logged before it runs
    with _holdout_access():
        result = runner()
    log = pd.read_parquet(ADDENDUM_LOG_PATH)
    row = log.index[log["test_id"] == test_id][-1]
    log.loc[row, "finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.loc[row, "summary"] = json.dumps(result.get("summary", {}), default=str)
    log.to_parquet(ADDENDUM_LOG_PATH, index=False)
    result["looks"] = previous + 1
    return result
