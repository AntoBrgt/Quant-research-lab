"""Triple-barrier trade labels: does a pick hit its target, its stop, or run out of time?

Why: the app's short-horizon use is a trading loop -- buy a pick, exit at the
stop or the target the Company Research page shows, or after N days -- not a
3-month cross-sectional ranking (STEP 11b/12 tested that; no edge). A trade
has a different question: *which barrier comes first*, after costs.

Rules, all conservative:
- **Entry at the next day's OPEN** after the signal close (the signal uses the
  signal day's close; trading at that same close would be optimistic). An
  entry that gaps to/below the stop or to/above the target is not taken
  (`outcome="skipped"`): the planned trade no longer exists.
- **Levels** come from the signal close, via `risk_reward.compute_risk_reward`
  (the app's own stop/target for the horizon) or, for the parameter grid,
  `stop = close - k x ATR`, `target = entry-risk x R`.
- **Walk daily bars.** A later day opening beyond a barrier exits at that open
  if it's the stop (a gap through the stop fills worse), at the target price if
  it's the target (no credit for a lucky gap). Within a bar, if both the stop
  and the target are touched, the STOP wins -- daily bars can't tell the order.
- **Time barrier**: exit at the close of the `max_hold`-th day (entry day = day 1).
  A ticker whose prices stop before that exits at its last close (`data_end`).
- **Costs**: `COST_PER_SIDE` + `SLIPPAGE_PER_SIDE` on entry AND exit.

Returns are net of costs; the R-multiple is the net return divided by the
planned risk (entry - stop) / entry. y = 1 only if the target came first.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import risk_reward

COST_PER_SIDE = 0.0010
SLIPPAGE_PER_SIDE = 0.0005
ATR_WINDOW = 14
SWING_WINDOW = 60

OUTCOMES = ("target", "stop", "time", "data_end", "skipped")


def side_cost() -> float:
    return COST_PER_SIDE + SLIPPAGE_PER_SIDE


def net_return(entry: np.ndarray, exit_: np.ndarray, cost: Optional[float] = None) -> np.ndarray:
    """Round trip after costs on both sides: buy at entry x (1 + c), sell at exit x (1 - c)."""
    c = side_cost() if cost is None else cost
    return exit_ * (1 - c) / (entry * (1 + c)) - 1


def aligned_ohlc(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """open/high/low/close on the benchmark calendar; NaN where the ticker didn't trade (never filled)."""
    frame = prices.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return pd.DataFrame({
        "open": frame["open"], "high": frame["high"], "low": frame["low"], "close": frame["adj_close"],
    }).reindex(calendar)


def level_inputs(prices: pd.DataFrame) -> pd.DataFrame:
    """ATR(14) and 60-day swing low/high per row, as of that row.

    Identical to `market_features.compute_atr` / `compute_swing_levels` with
    `as_of` = the row's date (tested), computed once per ticker instead of
    once per (date, ticker). Rolling windows only look backwards.
    """
    frame = prices.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    high, low, close = frame["high"], frame["low"], frame["adj_close"]
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(ATR_WINDOW).mean()
    atr[np.arange(len(frame)) < ATR_WINDOW] = np.nan  # compute_atr needs window + 1 rows
    return pd.DataFrame({
        "close": close,
        "atr_14d": atr,
        "support_60d": low.rolling(SWING_WINDOW).min(),
        "resistance_60d": high.rolling(SWING_WINDOW).max(),
    })


def app_levels(close: float, atr: Optional[float], support: Optional[float], resistance: Optional[float],
               horizon_days: int) -> tuple[Optional[float], Optional[float]]:
    """(stop, target) exactly as the app computes them for `horizon_days`."""
    clean = lambda v: None if v is None or pd.isna(v) else float(v)  # noqa: E731
    rr = risk_reward.compute_risk_reward(
        float(close), {"atr_14d": clean(atr), "support_60d": clean(support), "resistance_60d": clean(resistance)},
        None, horizon_days,
    )
    return rr["stop_loss"].get("level"), rr["take_profit"].get("level")


def grid_levels(close: np.ndarray, atr: np.ndarray, stop_atr: float, target_r: float) -> tuple[np.ndarray, np.ndarray]:
    stop = close - stop_atr * atr
    return stop, close + target_r * (close - stop)


def walk_barriers(
    ohlc: pd.DataFrame, signal_pos: np.ndarray, stop: np.ndarray, target: np.ndarray, max_hold: int,
    cost: Optional[float] = None,
) -> pd.DataFrame:
    """Vectorized first-touch for many trades on ONE ticker.

    `ohlc` is `aligned_ohlc` output, `signal_pos` integer positions of the
    signal dates in it; entry is at `signal_pos + 1`. Returns one row per
    signal: outcome, entry/exit position and price, days_held, return (net),
    r_multiple, y.
    """
    o, h, l, c = (ohlc[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close"))
    n_bars = len(o)
    signal_pos = np.asarray(signal_pos, dtype=int)
    stop = np.asarray(stop, dtype=float)
    target = np.asarray(target, dtype=float)
    k = np.arange(max_hold)
    idx = signal_pos[:, None] + 1 + k[None, :]
    inside = idx < n_bars
    safe = np.where(inside, idx, 0)
    O, H, L, C = (np.where(inside, a[safe], np.nan) for a in (o, h, l, c))

    entry = O[:, 0]
    valid = np.isfinite(entry) & np.isfinite(stop) & np.isfinite(target) & (stop > 0) & (entry > stop) & (entry < target)

    gap_stop = np.zeros_like(O, dtype=bool)
    gap_stop[:, 1:] = O[:, 1:] <= stop[:, None]
    gap_target = np.zeros_like(O, dtype=bool)
    gap_target[:, 1:] = O[:, 1:] >= target[:, None]
    stop_hit = gap_stop | (L <= stop[:, None])
    target_hit = (gap_target | (H >= target[:, None])) & ~stop_hit  # same bar: stop wins
    event = stop_hit | target_hit
    missing = ~np.isfinite(C)

    first_event = np.where(event.any(axis=1), event.argmax(axis=1), max_hold)
    first_missing = np.where(missing.any(axis=1), missing.argmax(axis=1), max_hold)
    rows = np.arange(len(signal_pos))

    outcome = np.full(len(signal_pos), "time", dtype=object)
    exit_k = np.full(len(signal_pos), max_hold - 1)
    exit_price = C[rows, max_hold - 1]

    by_event = first_event < first_missing
    is_stop = by_event & stop_hit[rows, np.minimum(first_event, max_hold - 1)]
    is_target = by_event & ~is_stop
    ek = np.minimum(first_event, max_hold - 1)
    outcome[is_stop] = "stop"
    outcome[is_target] = "target"
    exit_k = np.where(by_event, ek, exit_k)
    stop_fill = np.where(gap_stop[rows, ek], O[rows, ek], stop)
    exit_price = np.where(is_stop, stop_fill, np.where(is_target, target, exit_price))

    data_end = ~by_event & (first_missing < max_hold)
    last_ok = np.maximum(first_missing - 1, 0)
    outcome[data_end] = "data_end"
    exit_k = np.where(data_end, last_ok, exit_k)
    exit_price = np.where(data_end, C[rows, last_ok], exit_price)
    outcome[~valid] = "skipped"

    ret = np.where(valid, net_return(entry, exit_price, cost), np.nan)
    risk = (entry - stop) / entry
    return pd.DataFrame({
        "outcome": outcome,
        "entry_pos": signal_pos + 1,
        "exit_pos": signal_pos + 1 + exit_k,
        "entry_price": entry,
        "exit_price": np.where(valid, exit_price, np.nan),
        "stop": stop,
        "target": target,
        "days_held": np.where(valid, exit_k + 1, np.nan),
        "return": ret,
        "r_multiple": np.where(valid, ret / risk, np.nan),
        "y": np.where(valid, (outcome == "target").astype(float), np.nan),
    })
