"""Pure-Python market/technical features. No LLM calls, ever.

Every function accepts an explicit `as_of` cutoff and computes using only rows
with `date <= as_of` (default: the latest available date). This is what keeps
the research engine and backtests free of look-ahead bias -- a feature
computed "as of" a given date must be identical no matter what happens after
that date, which is exactly what `test_lookahead_bias.py` checks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _as_of_slice(prices: pd.DataFrame, as_of: Optional[pd.Timestamp]) -> pd.DataFrame:
    prices = prices.sort_index()
    if as_of is None:
        return prices
    return prices.loc[prices.index <= pd.Timestamp(as_of)]


def compute_returns(prices: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, window: int = 21) -> Optional[float]:
    """Simple return over the trailing `window` trading days, as of `as_of`."""
    window_df = _as_of_slice(prices, as_of)
    if len(window_df) < window + 1:
        return None
    closes = window_df["adj_close"]
    return float(closes.iloc[-1] / closes.iloc[-1 - window] - 1)


def compute_volatility(prices: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, window: int = 21) -> Optional[float]:
    """Annualized volatility of daily returns over the trailing `window` days."""
    window_df = _as_of_slice(prices, as_of)
    if len(window_df) < window + 1:
        return None
    daily_returns = window_df["adj_close"].pct_change().dropna().tail(window)
    if len(daily_returns) < 2:
        return None
    return float(daily_returns.std() * np.sqrt(252))


def compute_moving_average(prices: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, window: int = 50) -> Optional[float]:
    window_df = _as_of_slice(prices, as_of)
    if len(window_df) < window:
        return None
    return float(window_df["adj_close"].tail(window).mean())


def compute_rsi(prices: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, window: int = 14) -> Optional[float]:
    """Standard Wilder RSI over the trailing `window` days."""
    window_df = _as_of_slice(prices, as_of)
    if len(window_df) < window + 1:
        return None

    delta = window_df["adj_close"].diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.tail(window).mean()
    avg_loss = losses.tail(window).mean()

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_volume_ratio(prices: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, window: int = 21) -> Optional[float]:
    """Latest volume relative to its trailing `window`-day average."""
    window_df = _as_of_slice(prices, as_of)
    if "volume" not in window_df.columns or len(window_df) < window + 1:
        return None
    avg_volume = window_df["volume"].tail(window + 1).iloc[:-1].mean()
    if not avg_volume:
        return None
    return float(window_df["volume"].iloc[-1] / avg_volume)


def compute_features(prices: pd.DataFrame, as_of: Optional[pd.Timestamp] = None) -> dict:
    """Bundle of all features as of a given date -- the shape research_engine consumes."""
    return {
        "as_of": str(pd.Timestamp(as_of)) if as_of is not None else (
            str(prices.index.max()) if not prices.empty else None
        ),
        "return_1m": compute_returns(prices, as_of, window=21),
        "return_3m": compute_returns(prices, as_of, window=63),
        "volatility_1m": compute_volatility(prices, as_of, window=21),
        "moving_average_50d": compute_moving_average(prices, as_of, window=50),
        "moving_average_200d": compute_moving_average(prices, as_of, window=200),
        "rsi_14d": compute_rsi(prices, as_of, window=14),
        "volume_ratio": compute_volume_ratio(prices, as_of, window=21),
    }
