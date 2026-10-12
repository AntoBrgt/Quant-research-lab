"""PHASE 2: event-driven backtest, as fixed in PREREGISTRATION_PHASE2.md.

One engine for every hypothesis. An event is (signal_date, ticker, strength):
the day information became public and how strong it was. Everything else --
entry, exits, costs, benchmark, random baseline, verdict -- is shared, so the
three hypotheses are judged by exactly the same rules.

Conventions (all from the pre-registration):
- entry at the OPEN of the first trading day strictly after the signal date;
- exits via `barriers.walk_barriers` (time + optional ATR stop) or
  `value_survival.trade_outcome` (H3's target / thesis break / cap);
- excess = net trade return - benchmark return from the same entry open to the
  same exit day's close;
- development trades must EXIT by DEV_END; the holdout is run once (lock file).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest import barriers
from backtest.evaluate import newey_west_tstat
from backtest.strategy import exposure_matched_benchmark, max_drawdown

DEV_START = pd.Timestamp("2013-09-01")
DEV_END = pd.Timestamp("2024-08-31")
HOLDOUT_START = pd.Timestamp("2024-09-01")
MAX_OPEN = 20
RANDOM_SEEDS = 200
T_BAR = 2.4
RANDOM_PCT_BAR = 95.0
BOOT_REPS = 2000
FAR_TARGET = 1e12  # "no target": walk_barriers needs a finite level above entry
NO_STOP = 1e-12    # "no stop": positive, below any price

TRADE_COLUMNS = ["signal_date", "ticker", "strength", "entry_pos", "exit_pos", "entry_date", "exit_date",
                 "entry_price", "exit_price", "outcome", "ret", "bench_ret", "excess"]


# ----------------------------------------------------------------------------
# Exit rules
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class BarrierExit:
    """Time exit after `hold` days, optional stop at signal close - stop_atr x ATR(14)."""
    hold: int
    stop_atr: Optional[float]
    cost: float


@dataclass(frozen=True)
class OutcomeExit:
    """H3: `value_survival.trade_outcome` at one target (target / thesis break / cap / data end)."""
    target: float
    cost: float
    cap: int
    breaks: dict  # ticker -> list of thesis-break filing dates


def signal_positions(calendar: pd.DatetimeIndex, signal_dates) -> np.ndarray:
    """Position of the last trading day <= signal date; entry is the NEXT position."""
    return calendar.searchsorted(pd.DatetimeIndex(signal_dates), side="right") - 1


def _barrier_trades(ticker: str, frame: pd.DataFrame, ohlc: pd.DataFrame, levels: pd.DataFrame,
                    rule: BarrierExit) -> pd.DataFrame:
    calendar = ohlc.index
    pos = signal_positions(calendar, frame["signal_date"])
    ok = pos >= 0
    frame, pos = frame[ok], pos[ok]
    if frame.empty:
        return pd.DataFrame()
    if rule.stop_atr is None:
        stop = np.full(len(pos), NO_STOP)
    else:
        close = levels["close"].reindex(calendar).to_numpy(dtype=float)[pos]
        atr = levels["atr_14d"].reindex(calendar).to_numpy(dtype=float)[pos]
        stop = close - rule.stop_atr * atr
    walked = barriers.walk_barriers(ohlc, pos, stop, np.full(len(pos), FAR_TARGET), rule.hold, cost=rule.cost)
    walked.index = frame.index
    return pd.DataFrame({
        "signal_date": frame["signal_date"].to_numpy(),
        "ticker": ticker,
        "strength": frame["strength"].to_numpy(),
        "entry_pos": walked["entry_pos"].to_numpy(),
        "exit_pos": walked["exit_pos"].to_numpy(),
        "entry_price": walked["entry_price"].to_numpy(),
        "exit_price": walked["exit_price"].to_numpy(),
        "outcome": walked["outcome"].to_numpy(),
        "ret": walked["return"].to_numpy(),
    })


def _outcome_trades(ticker: str, frame: pd.DataFrame, ohlc: pd.DataFrame, rule: OutcomeExit) -> pd.DataFrame:
    from backtest import value_survival as vs

    key = f"t{int(round(rule.target * 100))}"
    calendar = ohlc.index
    rows = []
    for signal_date, strength in zip(frame["signal_date"], frame["strength"]):
        out = vs.trade_outcome(ohlc, pd.Timestamp(signal_date), targets=(rule.target,),
                               break_dates=rule.breaks.get(ticker, ()), cap=rule.cap, cost=rule.cost)
        if out.get("entry_date") is None:
            continue
        rows.append({
            "signal_date": pd.Timestamp(signal_date), "ticker": ticker, "strength": strength,
            "entry_pos": calendar.get_loc(out["entry_date"]), "exit_pos": calendar.get_loc(out[f"{key}_exit_date"]),
            "entry_price": out["entry_price"], "exit_price": out[f"{key}_exit_price"],
            "outcome": out[f"{key}_reason"], "ret": out[f"{key}_return"],
        })
    return pd.DataFrame(rows)


def build_trades(events: pd.DataFrame, ohlc: dict[str, pd.DataFrame], bench_ohlc: pd.DataFrame,
                 rule, levels: Optional[dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """events (signal_date, ticker, strength) -> one row per taken trade with its excess return.

    `ohlc` values must be `barriers.aligned_ohlc` on the benchmark's calendar.
    Skipped entries (no price, gap through the stop) are dropped.
    """
    parts = []
    for ticker, frame in events.groupby("ticker", sort=True):
        if ticker not in ohlc:
            continue
        frame = frame.sort_values("signal_date")
        if isinstance(rule, BarrierExit):
            lv = levels[ticker] if levels is not None and ticker in levels else barriers.level_inputs(_as_prices(ohlc[ticker]))
            part = _barrier_trades(ticker, frame, ohlc[ticker], lv, rule)
        else:
            part = _outcome_trades(ticker, frame, ohlc[ticker], rule)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    trades = pd.concat(parts, ignore_index=True)
    trades = trades[trades["outcome"] != "skipped"].dropna(subset=["ret"]).reset_index(drop=True)
    calendar = bench_ohlc.index
    b_open = bench_ohlc["open"].to_numpy(dtype=float)
    b_close = bench_ohlc["close"].to_numpy(dtype=float)
    ep, xp = trades["entry_pos"].to_numpy(dtype=int), trades["exit_pos"].to_numpy(dtype=int)
    trades["entry_date"] = calendar[ep]
    trades["exit_date"] = calendar[xp]
    trades["bench_ret"] = b_close[xp] / b_open[ep] - 1
    trades["excess"] = trades["ret"] - trades["bench_ret"]
    return trades[TRADE_COLUMNS].dropna(subset=["excess"]).reset_index(drop=True)


def _as_prices(ohlc: pd.DataFrame) -> pd.DataFrame:
    """aligned_ohlc -> the column names level_inputs expects (rows where it traded)."""
    frame = ohlc.dropna(subset=["close"])
    return frame.rename(columns={"close": "adj_close"})


# ----------------------------------------------------------------------------
# Periods
# ----------------------------------------------------------------------------


def period_trades(trades: pd.DataFrame, period: str) -> pd.DataFrame:
    s = pd.to_datetime(trades["signal_date"])
    if period == "development":
        mask = (s >= DEV_START) & (s <= DEV_END) & (pd.to_datetime(trades["exit_date"]) <= DEV_END)
    elif period == "holdout":
        mask = s >= HOLDOUT_START
    else:
        raise ValueError(period)
    return trades[mask].reset_index(drop=True)


def period_events(events: pd.DataFrame, period: str) -> pd.DataFrame:
    """Events whose signal date falls in the period (the dev exit rule is applied to trades)."""
    s = pd.to_datetime(events["signal_date"])
    mask = (s >= DEV_START) & (s <= DEV_END) if period == "development" else s >= HOLDOUT_START
    return events[mask].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------


def monthly_excess(trades: pd.DataFrame) -> pd.Series:
    month = pd.to_datetime(trades["entry_date"]).dt.to_period("M")
    return trades.groupby(month)["excess"].mean().sort_index()


def nw_tstat(trades: pd.DataFrame, hold: int) -> Optional[float]:
    series = monthly_excess(trades)
    return newey_west_tstat(series.to_numpy(), lags=max(1, math.ceil(hold / 21)))


def bootstrap_ci(trades: pd.DataFrame, reps: int = BOOT_REPS, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """CI of the mean excess per trade, resampling entry MONTHS (trades in a month are correlated)."""
    if trades.empty:
        return (float("nan"), float("nan"))
    month = pd.to_datetime(trades["entry_date"]).dt.to_period("M")
    sums = trades.groupby(month)["excess"].sum().to_numpy()
    counts = trades.groupby(month)["excess"].size().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(reps, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


class Eligibility:
    """Point-in-time universe: on date d, the members listed at the latest membership date <= d."""

    def __init__(self, members: pd.DataFrame):
        frame = members[["date", "ticker"]].dropna().copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self.dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        self.pools = {d: sorted(g["ticker"].unique()) for d, g in frame.groupby("date")}
        self.sets = {d: set(p) for d, p in self.pools.items()}

    def as_of(self, date) -> Optional[pd.Timestamp]:
        i = self.dates.searchsorted(pd.Timestamp(date), side="right") - 1
        return self.dates[i] if i >= 0 else None

    def pool(self, date) -> list[str]:
        key = self.as_of(date)
        return self.pools.get(key, []) if key is not None else []

    def contains(self, date, ticker: str) -> bool:
        key = self.as_of(date)
        return key is not None and ticker in self.sets[key]

    def filter(self, events: pd.DataFrame) -> pd.DataFrame:
        keep = [self.contains(d, t) for d, t in zip(events["signal_date"], events["ticker"])]
        return events[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def random_events(trades: pd.DataFrame, eligible: Eligibility, seed: int) -> pd.DataFrame:
    """Each real trade's signal date with a random ticker from that date's eligible universe."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.to_datetime(trades["signal_date"]):
        pool = eligible.pool(d)
        if not pool:
            continue
        rows.append({"signal_date": d, "ticker": pool[int(rng.integers(len(pool)))], "strength": 0.0})
    return pd.DataFrame(rows, columns=["signal_date", "ticker", "strength"])


def random_baseline(trades: pd.DataFrame, eligible: "Eligibility", ohlc: dict, bench_ohlc: pd.DataFrame, rule,
                    levels: Optional[dict] = None, seeds: int = RANDOM_SEEDS, period: Optional[str] = None) -> np.ndarray:
    """Mean excess of `seeds` random pickers with the same dates and exits."""
    means = []
    for seed in range(seeds):
        fake = build_trades(random_events(trades, eligible, seed), ohlc, bench_ohlc, rule, levels)
        if period is not None:
            fake = period_trades(fake, period)
        means.append(float(fake["excess"].mean()) if len(fake) else np.nan)
    return np.array(means)


def percentile_of(value: float, distribution: np.ndarray) -> float:
    dist = distribution[np.isfinite(distribution)]
    return float((dist < value).mean() * 100) if len(dist) else float("nan")


# ----------------------------------------------------------------------------
# Portfolio
# ----------------------------------------------------------------------------


def simulate_portfolio(trades: pd.DataFrame, ohlc: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex,
                       cost: float, max_open: int = MAX_OPEN) -> dict:
    """Daily equity: <= max_open positions, 1/max_open of equity each at entry, idle cash at 0%.

    Same-day entries are taken by strength (desc), then ticker. Positions are
    marked at the forward-filled close; exits at the trade's exit price.
    """
    if trades.empty:
        return {"equity": pd.Series(dtype=float), "exposure": pd.Series(dtype=float), "taken": trades}
    order = trades.sort_values(["entry_pos", "strength", "ticker"], ascending=[True, False, True]).reset_index(drop=True)
    closes = {t: ohlc[t]["close"].ffill().to_numpy(dtype=float) for t in order["ticker"].unique()}
    start, end = int(order["entry_pos"].min()), int(order["exit_pos"].max())
    by_entry = {p: g for p, g in order.groupby("entry_pos")}
    cash, open_pos, taken = 1.0, [], []
    equity, exposure = [], []
    for day in range(start, end + 1):
        # exits first (a slot freed today can't be refilled at today's open: exits happen during/after)
        prev_value = cash + sum(p["units"] * closes[p["ticker"]][day - 1] for p in open_pos) if day > start else cash
        if day in by_entry:
            for r in by_entry[day].itertuples(index=False):
                if len(open_pos) >= max_open:
                    continue
                alloc = min(prev_value / max_open, cash)
                if alloc <= 0:
                    continue
                cash -= alloc
                open_pos.append({"ticker": r.ticker, "units": alloc / (r.entry_price * (1 + cost)),
                                 "exit_pos": int(r.exit_pos), "exit_price": float(r.exit_price)})
                taken.append(r)
        still = []
        for p in open_pos:
            if p["exit_pos"] == day:
                cash += p["units"] * p["exit_price"] * (1 - cost)
            else:
                still.append(p)
        open_pos = still
        invested = sum(p["units"] * closes[p["ticker"]][day] for p in open_pos)
        total = cash + invested
        equity.append(total)
        exposure.append(invested / total if total > 0 else 0.0)
    index = calendar[start:end + 1]
    return {"equity": pd.Series(equity, index=index), "exposure": pd.Series(exposure, index=index),
            "taken": pd.DataFrame(taken)}


def portfolio_metrics(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {}
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    daily = equity.pct_change().dropna()
    return {"total_return": total, "cagr": (1 + total) ** (1 / years) - 1 if years > 0 else float("nan"),
            "max_drawdown": max_drawdown(equity),
            "sharpe": float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else float("nan")}


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------


def evaluate_period(trades: pd.DataFrame, hold: int, random_means: np.ndarray, sim: dict, bench_close: pd.Series) -> dict:
    ci = bootstrap_ci(trades)
    t = nw_tstat(trades, hold) if len(trades) else None
    pm = portfolio_metrics(sim["equity"])
    bm = exposure_matched_benchmark(bench_close, sim["exposure"]) if len(sim["exposure"]) else {}
    mean = float(trades["excess"].mean()) if len(trades) else float("nan")
    pct = percentile_of(mean, random_means)
    checks = {
        "mean_excess_positive": bool(mean > 0),
        "t_at_least_2.4": bool(t is not None and t >= T_BAR),
        "ci_above_zero": bool(np.isfinite(ci[0]) and ci[0] > 0),
        "beats_95pct_random": bool(np.isfinite(pct) and pct >= RANDOM_PCT_BAR),
        "portfolio_beats_benchmark": bool(pm and bm and pm["cagr"] > bm["cagr"]),
    }
    return {
        "trades": int(len(trades)), "months": int(monthly_excess(trades).size) if len(trades) else 0,
        "mean_return": float(trades["ret"].mean()) if len(trades) else float("nan"),
        "mean_excess": mean, "win_rate": float((trades["ret"] > 0).mean()) if len(trades) else float("nan"),
        "nw_t": t, "ci_low": ci[0], "ci_high": ci[1], "random_pct": pct,
        "portfolio": pm, "benchmark_same_exposure": bm, "checks": checks,
        "passed": all(checks.values()),
    }


def verdict(dev: dict, holdout: Optional[dict]) -> str:
    if holdout is None:
        return "DEVELOPMENT ONLY (holdout not run)"
    return "PASS" if dev["passed"] and holdout["passed"] else "NOT PROVEN"


# ----------------------------------------------------------------------------
# Holdout lock
# ----------------------------------------------------------------------------


def holdout_lock(out_dir: Path, commit: str, force: bool) -> dict:
    """Allow ONE holdout run. Returns the lock record; raises if already run and not forced."""
    path = Path(out_dir) / "holdout.lock"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if previous and not force:
        raise RuntimeError(f"Holdout already run at {previous['runs'][0]['time']} (commit {previous['runs'][0]['commit']}). "
                           "Re-running it would make it a second development set. Use --force only to reproduce, "
                           "and the report will say so.")
    record = previous or {"runs": []}
    record["runs"].append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit, "forced": bool(previous)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
