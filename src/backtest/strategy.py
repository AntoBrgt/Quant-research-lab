"""Trade-level strategy backtest: buy the week's picks, exit at stop/target/time, repeat.

Why: the ranking tests (STEP 11b/12) ask whether the score orders 3-month
returns. The app is used differently -- as a short-term trading list with a
stop and a target per name. That loop can make money even with a weak
ranking signal (or lose it with a decent one: costs, stop placement,
position limits), so it is tested as a loop, on the point-in-time universe.

    each week (signal = that day's close):
        rank the PIT universe members by the variant's signal, take the top K
        enter at the next day's open (barriers.py), max P positions open,
        risk RISK_PER_TRADE of equity per trade (capped at MAX_POSITION_WEIGHT),
        exit at the first barrier: stop / target / max hold days

Variants: `score_rank` (the app's short-horizon rank_score, with the Quick
Picks filter: Strong/Favorable and app risk/reward >= 1), `momentum_12_1`,
`breakout_20d` (close above the prior 20-day high, ranked by 20-day return).

**Random-entry baseline**: the same exits, sizing and position limits with
random picks from the same week's universe, over many seeds. If the strategy
doesn't beat ~95% of random pickers, its result is what the exits and the
market did, not what the picks did.

Picks are chosen before the next open is known: a pick whose entry would gap
through its stop or target is simply not filled (no replacement from further
down the list -- that would use the next day's open to choose).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest import barriers

logger = logging.getLogger(__name__)

TOP_K = 5
MAX_POSITIONS = 10
RISK_PER_TRADE = 0.01
MAX_POSITION_WEIGHT = 0.20
REBALANCE_EVERY = 5  # trading days: weekly
SHORT_HORIZON_DAYS = 14  # the app's "2 weeks" horizon: scoring profile + risk_reward category
MAX_HOLDS = (5, 10, 20)
MAIN_HOLD = 10
GRID_STOP_ATR = (1.5, 2.0, 3.0)
GRID_TARGET_R = (1.5, 2.0, 3.0)
RANDOM_SEEDS = 100
BREAKOUT_WINDOW = 20
FAVORABLE_THRESHOLD = 0.10  # screener.FAVORABLE_THRESHOLD: Strong/Favorable labels
MIN_QUICK_PICKS_RR = 1.0  # Quick Picks only buys when the target is at least as far as the stop


# ----------------------------------------------------------------------------
# Candidates and barrier outcomes
# ----------------------------------------------------------------------------


def build_candidates(
    panel: pd.DataFrame, prices: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, horizon_days: int = SHORT_HORIZON_DAYS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """One row per (signal date, ticker) PIT member with a close that day.

    Adds the calendar position of the signal, ATR / swing levels as of the
    signal close, the app's stop/target/risk-reward for `horizon_days`, and the
    20-day breakout flag. Returns (candidates, aligned OHLC per ticker).
    """
    calendar_pos = pd.Series(np.arange(len(calendar)), index=calendar)
    frames, ohlc = [], {}
    for ticker, group in panel.groupby("ticker"):
        if ticker not in prices:
            continue
        raw = prices[ticker].sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]
        levels = barriers.level_inputs(raw)
        levels["prior_high_20d"] = raw["high"].shift(1).rolling(BREAKOUT_WINDOW).max()
        rows = group.merge(levels, left_on="date", right_index=True, how="inner")
        rows = rows[rows["date"].isin(calendar_pos.index)]
        if rows.empty:
            continue
        rows = rows.assign(signal_pos=calendar_pos.loc[rows["date"]].to_numpy())
        frames.append(rows)
        ohlc[ticker] = barriers.aligned_ohlc(raw, calendar)
    candidates = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)

    levels = [barriers.app_levels(r.close, r.atr_14d, r.support_60d, r.resistance_60d, horizon_days)
              for r in candidates[["close", "atr_14d", "support_60d", "resistance_60d"]].itertuples(index=False)]
    candidates["app_stop"] = [s for s, _ in levels]
    candidates["app_target"] = [t for _, t in levels]
    risk = candidates["close"] - candidates["app_stop"]
    candidates["app_rr"] = (candidates["app_target"] - candidates["close"]) / risk.where(risk > 0)
    candidates["breakout_20d"] = candidates["close"] > candidates["prior_high_20d"]
    candidates["stop_dist_pct"] = risk / candidates["close"]
    candidates["target_dist_pct"] = (candidates["app_target"] - candidates["close"]) / candidates["close"]
    candidates["atr_pct"] = candidates["atr_14d"] / candidates["close"]
    return candidates, ohlc


@dataclass(frozen=True)
class Geometry:
    """How stop/target are set: the app's own levels, or a grid cell."""
    kind: str  # "app", "grid" (fixed stop/target) or "trail" (trailing stop, no target)
    max_hold: int
    stop_atr: Optional[float] = None
    target_r: Optional[float] = None

    @property
    def key(self) -> str:
        if self.kind == "app":
            return f"app_hold{self.max_hold}"
        if self.kind == "trail":
            return f"trail{self.stop_atr:g}atr_hold{self.max_hold}"
        return f"atr{self.stop_atr:g}_r{self.target_r:g}_hold{self.max_hold}"


def outcomes(candidates: pd.DataFrame, ohlc: dict[str, pd.DataFrame], geometry: Geometry) -> pd.DataFrame:
    """Barrier outcome for every candidate under `geometry`, index-aligned with `candidates`."""
    parts = []
    for ticker, rows in candidates.groupby("ticker"):
        if geometry.kind == "trail":
            result = barriers.walk_trailing_stop(ohlc[ticker], rows["signal_pos"].to_numpy(),
                                                 rows["atr_14d"].to_numpy(dtype=float), geometry.stop_atr, geometry.max_hold)
            result.index = rows.index
            parts.append(result)
            continue
        if geometry.kind == "app":
            stop, target = rows["app_stop"].to_numpy(dtype=float), rows["app_target"].to_numpy(dtype=float)
        else:
            stop, target = barriers.grid_levels(rows["close"].to_numpy(dtype=float), rows["atr_14d"].to_numpy(dtype=float),
                                                geometry.stop_atr, geometry.target_r)
        result = barriers.walk_barriers(ohlc[ticker], rows["signal_pos"].to_numpy(), stop, target, geometry.max_hold)
        result.index = rows.index
        parts.append(result)
    return pd.concat(parts).reindex(candidates.index)


# ----------------------------------------------------------------------------
# Pick rules
# ----------------------------------------------------------------------------

PickRule = Callable[[pd.DataFrame], pd.DataFrame]


def pick_score_rank(week: pd.DataFrame, geometry_is_app: bool = True) -> pd.DataFrame:
    """The app's Quick Picks: Strong/Favorable by rank_score, risk/reward >= 1, best first."""
    eligible = week[week["score_rank"] >= FAVORABLE_THRESHOLD]
    if geometry_is_app:
        eligible = eligible[eligible["app_rr"] >= MIN_QUICK_PICKS_RR]
    return eligible.sort_values("score_rank", ascending=False)


def pick_momentum(week: pd.DataFrame, geometry_is_app: bool = True) -> pd.DataFrame:
    return week.dropna(subset=["momentum_12_1"]).sort_values("momentum_12_1", ascending=False)


def pick_breakout(week: pd.DataFrame, geometry_is_app: bool = True) -> pd.DataFrame:
    return week[week["breakout_20d"]].sort_values("return_20d", ascending=False)


PICK_RULES = {"score_rank": pick_score_rank, "momentum_12_1": pick_momentum, "breakout_20d": pick_breakout}


def weekly_picks(candidates: pd.DataFrame, rule: Callable, k: int = TOP_K, geometry_is_app: bool = True,
                 mask: Optional[pd.Series] = None) -> dict[pd.Timestamp, list[int]]:
    """{signal date: candidate indices, best first} -- chosen from the signal close only."""
    pool = candidates if mask is None else candidates[mask]
    return {date: rule(week, geometry_is_app).index[:k].tolist() for date, week in pool.groupby("date")}


def random_picks(candidates: pd.DataFrame, seed: int, k: int = TOP_K, mask: Optional[pd.Series] = None) -> dict[pd.Timestamp, list[int]]:
    """K random members per week (same universe), reproducible by seed."""
    rng = np.random.default_rng(seed)
    pool = candidates if mask is None else candidates[mask]
    return {date: rng.permutation(week.index.to_numpy())[:k].tolist() for date, week in pool.groupby("date")}


# ----------------------------------------------------------------------------
# Portfolio simulation
# ----------------------------------------------------------------------------


@dataclass
class SimResult:
    equity: pd.Series
    trades: pd.DataFrame
    skipped: int = 0
    not_opened: int = 0
    meta: dict = field(default_factory=dict)
    exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # invested value / equity, at each close


def simulate(
    candidates: pd.DataFrame,
    result: pd.DataFrame,
    picks: dict[pd.Timestamp, list[int]],
    closes: dict[str, np.ndarray],
    calendar: pd.DatetimeIndex,
    max_positions: int = MAX_POSITIONS,
    risk_per_trade: float = RISK_PER_TRADE,
    max_weight: float = MAX_POSITION_WEIGHT,
    cost: Optional[float] = None,
) -> SimResult:
    """Day-by-day portfolio: entries at the open, barrier exits, mark-to-market at the close.

    Sizing uses the previous close's equity: position value = equity x
    risk_per_trade / (planned risk fraction), capped at max_weight x equity and
    at available cash (no leverage). A pick already held, or beyond the
    position limit, is not opened (`not_opened`).
    """
    c = barriers.side_cost() if cost is None else cost
    entries: dict[int, list[int]] = {}
    for date, idxs in picks.items():
        for i in idxs:
            entries.setdefault(int(candidates.at[i, "signal_pos"]) + 1, []).append(i)
    if not entries:
        return SimResult(pd.Series(dtype=float), pd.DataFrame())

    start = min(entries) - 1
    cash, equity_prev = 1.0, 1.0
    open_positions: list[dict] = []
    trades, equity, exposure = [], [], []
    skipped = not_opened = 0
    for t in range(start, len(calendar)):
        for i in entries.get(t, []):
            row = result.loc[i]
            if row["outcome"] == "skipped":
                skipped += 1
                continue
            ticker = candidates.at[i, "ticker"]
            if len(open_positions) >= max_positions or any(p["ticker"] == ticker for p in open_positions):
                not_opened += 1
                continue
            entry = float(row["entry_price"])
            risk_frac = (entry - float(row["stop"])) / entry
            value = min(equity_prev * risk_per_trade / risk_frac, equity_prev * max_weight, cash / (1 + c))
            if value <= 0:
                not_opened += 1
                continue
            shares = value / entry
            cash -= shares * entry * (1 + c)
            open_positions.append({"i": i, "ticker": ticker, "shares": shares, "exit_pos": int(row["exit_pos"]),
                                   "exit_price": float(row["exit_price"]), "cost_basis": shares * entry * (1 + c),
                                   "weight": value / equity_prev})
        still_open = []
        for p in open_positions:
            if p["exit_pos"] == t:
                proceeds = p["shares"] * p["exit_price"] * (1 - c)
                cash += proceeds
                row = result.loc[p["i"]]
                trades.append({
                    "candidate": p["i"], "ticker": p["ticker"], "signal_date": candidates.at[p["i"], "date"],
                    "entry_date": calendar[int(row["entry_pos"])], "exit_date": calendar[t],
                    "outcome": row["outcome"], "days_held": row["days_held"], "return": row["return"],
                    "r_multiple": row["r_multiple"], "pnl": proceeds - p["cost_basis"],
                    "weight": p["weight"],  # position value / equity at entry
                })
            else:
                still_open.append(p)
        open_positions = still_open
        invested = 0.0
        for p in open_positions:
            px = closes[p["ticker"]][t]
            invested += p["shares"] * (px if np.isfinite(px) else p["exit_price"])
        marked = cash + invested
        equity.append(marked)
        exposure.append(invested / marked if marked > 0 else 0.0)
        equity_prev = marked
    index = calendar[start:]
    return SimResult(pd.Series(equity, index=index), pd.DataFrame(trades), skipped, not_opened,
                     exposure=pd.Series(exposure, index=index))


def forward_filled_closes(ohlc: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Closes for mark-to-market only (a halted day keeps its last close); never used for signals or fills."""
    return {t: frame["close"].ffill().to_numpy(dtype=float) for t, frame in ohlc.items()}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min()) if len(equity) else float("nan")


def metrics(sim: SimResult) -> dict:
    eq, trades = sim.equity, sim.trades
    if eq.empty:
        return {"trades": 0}
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1)
    daily = eq.pct_change().dropna()
    out = {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if total > -1 else -1.0,
        "max_drawdown": max_drawdown(eq),
        "sharpe": float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else float("nan"),
        "trades": len(trades),
    }
    if len(trades):
        wins, losses = trades[trades["return"] > 0], trades[trades["return"] <= 0]
        gross_win, gross_loss = wins["pnl"].sum(), -losses["pnl"].sum()
        out.update({
            "win_rate": len(wins) / len(trades),
            "avg_win": wins["return"].mean() if len(wins) else float("nan"),
            "avg_loss": losses["return"].mean() if len(losses) else float("nan"),
            "expectancy_r": trades["r_multiple"].mean(),
            "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
            "target_share": (trades["outcome"] == "target").mean(),
            "stop_share": (trades["outcome"] == "stop").mean(),
        })
    return out


def by_year(sim: SimResult) -> pd.DataFrame:
    eq = sim.equity
    if eq.empty:
        return pd.DataFrame()
    yearly = eq.groupby(eq.index.year).agg(["first", "last"])
    prior_last = yearly["last"].shift(1).fillna(yearly["first"])
    out = pd.DataFrame({"return": yearly["last"] / prior_last - 1})
    if len(sim.trades):
        t = sim.trades.assign(year=pd.to_datetime(sim.trades["exit_date"]).dt.year)
        out["trades"] = t.groupby("year").size()
        out["expectancy_r"] = t.groupby("year")["r_multiple"].mean()
    out.index.name = "year"
    return out


def bootstrap_expectancy_ci(trades: pd.DataFrame, reps: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """95% CI of mean R per trade, resampling WEEKS (trades entered the same week are correlated)."""
    if trades is None or len(trades) < 2:
        return float("nan"), float("nan")
    weeks = pd.to_datetime(trades["signal_date"])
    groups = [g["r_multiple"].to_numpy() for _, g in trades.groupby(weeks)]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(reps, len(groups)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def percentile_of(value: float, distribution) -> float:
    dist = np.asarray([d for d in distribution if np.isfinite(d)])
    return float((dist < value).mean() * 100) if len(dist) else float("nan")


def buy_and_hold(closes: pd.Series, start, end) -> dict:
    s = closes.loc[start:end].dropna()
    if len(s) < 2:
        return {}
    eq = s / s.iloc[0]
    years = (s.index[-1] - s.index[0]).days / 365.25
    total = float(eq.iloc[-1] - 1)
    period = eq.pct_change().dropna()
    # Annualize by the series' own frequency: the equal-weight curve is weekly, SPY daily.
    periods_per_year = len(period) / years
    return {"total_return": total, "cagr": (1 + total) ** (1 / years) - 1, "max_drawdown": max_drawdown(eq),
            "sharpe": float(period.mean() / period.std() * math.sqrt(periods_per_year))}


def equal_weight_universe(candidates: pd.DataFrame, closes: dict[str, np.ndarray], calendar: pd.DatetimeIndex) -> pd.Series:
    """Equal weight in every PIT member, rebalanced at each signal date (close to next signal close)."""
    dates = sorted(candidates["date"].unique())
    pos = {d: i for i, d in enumerate(calendar)}
    value, curve = 1.0, {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        members = candidates.loc[candidates["date"] == d0, "ticker"]
        rets = [closes[t][pos[d1]] / closes[t][pos[d0]] - 1 for t in members if t in closes
                and np.isfinite(closes[t][pos[d0]]) and np.isfinite(closes[t][pos[d1]])]
        value *= 1 + (np.mean(rets) if rets else 0.0)
        curve[d1] = value
    return pd.Series(curve)


def exposure_matched_benchmark(bench_closes: pd.Series, exposure: pd.Series) -> dict:
    """SPY held at the strategy's AVERAGE exposure (rest in cash at 0%), same period.

    A loop that is 40% invested on average should be compared with 40% of SPY,
    not 100%: otherwise "it trails SPY" partly just says "it holds less stock".
    """
    if exposure.empty:
        return {}
    avg = float(exposure.mean())
    spy = bench_closes.loc[exposure.index[0]:exposure.index[-1]].dropna()
    scaled = (1 + avg * spy.pct_change().fillna(0.0)).cumprod()
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    total = float(scaled.iloc[-1] - 1)
    daily = scaled.pct_change().dropna()
    return {"avg_exposure": avg, "total_return": total, "cagr": (1 + total) ** (1 / years) - 1,
            "max_drawdown": max_drawdown(scaled), "sharpe": float(daily.mean() / daily.std() * math.sqrt(252))}
