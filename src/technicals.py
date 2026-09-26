"""Technical analysis: historical price observations, deterministic indicators,
trend/momentum/volatility/volume classification, relative strength, and
support/resistance. No LLM anywhere in this file.

Same three-layer shape as `fundamentals.py`, applied to price data instead of
financial statements:

1. **Provider**: reuses `price_provider.py` as-is (cached OHLCV, no new
   provider introduced). Relative strength additionally fetches a benchmark
   ticker through the exact same provider.
2. **Historical series** (`TechnicalObservation`): indicators are built as
   chronological, dated series over a trailing window (not just the latest
   value) by looping `market_features.py`'s existing `as_of`-parametrized
   functions -- reused directly, never reimplemented here.
3. **Derived output** (`TechnicalAnalysis`): the flat schema below, plus
   deterministic BULLISH/BEARISH/MIXED-style trend states. Descriptive only --
   there is no BUY/SELL output and no single opaque "technical score"
   anywhere in this module; every component stays independently readable.

Price convention: every indicator uses `adj_close` -- `price_provider.py`
downloads with `auto_adjust=True`, so `open`/`high`/`low`/`adj_close` are all
the SAME fully split/dividend-adjusted series (see that module's docstring).
There is no separate raw/unadjusted series kept anywhere in this project, so
there is nothing left to accidentally mix.

Honesty rules (same as `fundamentals.py`): a missing/insufficient-history
metric is `None`, never fabricated, never substituted with another horizon,
zero, or an average (section 15); a suspicious value is flagged in
`quality_flags`/`data_quality`, never silently dropped (section 13) -- and a
real, sharp market move (confirmed by an accompanying volume spike) is
labeled `VALID_EXTREME_MARKET_MOVE`, not treated the same as a data error.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import config
import market_features
import price_provider
from fundamentals import classify_trend as _classify_trend_generic  # reused, not reimplemented
from fundamentals import MetricObservation as _GenericObservation

logger = logging.getLogger(__name__)

TrendDirection = Literal["BULLISH", "BEARISH", "MIXED", "INSUFFICIENT_DATA"]
RsiState = Literal["OVERBOUGHT", "NEUTRAL", "OVERSOLD", "INSUFFICIENT_DATA"]
VolatilityRegime = Literal["LOW", "NORMAL", "HIGH", "INSUFFICIENT_DATA"]
VolumeTrend = Literal["INCREASING", "STABLE", "DECREASING", "INSUFFICIENT_DATA"]

DATA_SOURCE = "price_provider:yfinance"
# How much recent daily history to preserve as dated indicator series
# (section 11). ~3 months of trading days -- enough to see indicator
# trend/regime context without dumping the full 2-year cache into every
# result.
HISTORY_WINDOW_DAYS = 60

# Return horizons this project actually consumes downstream (horizon.py,
# risk_reward.py, strategy.py) plus the ones this brief explicitly asks for.
RETURN_WINDOWS: dict[str, int] = {
    "1d": 1, "5d": 5, "20d": 20, "60d": 60, "120d": 120, "252d": 252,
}


# ---------------------------------------------------------------------------
# Historical series
# ---------------------------------------------------------------------------

class TechnicalObservation(BaseModel):
    """One dated value for one technical series -- section 11/14 provenance."""

    metric: str
    value: Optional[float]
    date: str  # ISO trading date
    source: str
    retrieved_at: str
    quality_flags: list[str] = Field(default_factory=list)


class ReturnObservation(BaseModel):
    """One return horizon's full provenance (section 4) -- not just the number."""

    horizon: str  # e.g. "20d"
    start_date: Optional[str]
    end_date: Optional[str]
    value: Optional[float]
    complete: bool  # False when there wasn't enough history for this horizon


def _trading_dates(prices: pd.DataFrame, window_days: int) -> list[pd.Timestamp]:
    if prices.empty:
        return []
    return list(prices.sort_index().index[-window_days:])


def _series_over_history(
    prices: pd.DataFrame, metric: str, compute_fn, retrieved_at: str, window_days: int = HISTORY_WINDOW_DAYS, **kwargs
) -> list[TechnicalObservation]:
    """Build a dated series by calling an existing `market_features.py`
    `as_of`-parametrized function once per trailing trading day. Reuses the
    exact same function real-time scoring uses, so the historical series and
    the latest value can never silently disagree.
    """
    observations = []
    for as_of in _trading_dates(prices, window_days):
        value = compute_fn(prices, as_of=as_of, **kwargs)
        observations.append(
            TechnicalObservation(metric=metric, value=value, date=str(as_of.date()), source="computed", retrieved_at=retrieved_at)
        )
    return observations


# ---------------------------------------------------------------------------
# Data quality validation (section 13) -- flags, never silently drops/discards
# ---------------------------------------------------------------------------

# Documented, tunable heuristics -- not statistically fit.
_EXTREME_DAILY_RETURN_ABS = 0.20  # +/-20% in one day is unusual enough to check
_VOLUME_CONFIRMATION_RATIO = 1.5  # that day's volume vs its trailing 20d average


def validate_ohlcv(prices: pd.DataFrame) -> list[str]:
    """Row-level and structural checks on the raw OHLCV frame.

    Returns flags prefixed by severity -- `INVALID_DATA` (a structural
    impossibility: this is always wrong regardless of market conditions),
    `SUSPICIOUS_DATA` (unusual and not corroborated by volume -- worth a
    look, not necessarily wrong), or `VALID_EXTREME_MARKET_MOVE` (a large
    move that IS corroborated by a volume spike -- a real market event, not
    a data error). Nothing is dropped from `prices` because of this.
    """
    flags: list[str] = []
    if prices.empty:
        return flags

    df = prices.sort_index()

    if not prices.index.equals(df.index):
        flags.append("INVALID_DATA:dates:unsorted")

    duplicate_dates = df.index[df.index.duplicated()]
    for d in duplicate_dates:
        flags.append(f"INVALID_DATA:dates:{d.date()}:duplicate_date")

    if len(df) < 20:
        flags.append(f"INSUFFICIENT_HISTORY:only_{len(df)}_rows")

    if "adj_close" in df.columns:
        negative_price = df[df["adj_close"] < 0]
        for d in negative_price.index:
            flags.append(f"INVALID_DATA:price:{d.date()}:negative_price")
        missing_close = df["adj_close"].isna()
        for d in df.index[missing_close]:
            flags.append(f"INVALID_DATA:price:{d.date()}:missing_close")

    if {"high", "low"}.issubset(df.columns):
        bad_hl = df[df["high"] < df["low"]]
        for d in bad_hl.index:
            flags.append(f"INVALID_DATA:ohlc:{d.date()}:high_less_than_low")

        if "adj_close" in df.columns:
            close_above_high = df[df["adj_close"] > df["high"]]
            for d in close_above_high.index:
                flags.append(f"INVALID_DATA:ohlc:{d.date()}:close_above_high")
            close_below_low = df[df["adj_close"] < df["low"]]
            for d in close_below_low.index:
                flags.append(f"INVALID_DATA:ohlc:{d.date()}:close_below_low")

    if "volume" in df.columns:
        negative_volume = df[df["volume"] < 0]
        for d in negative_volume.index:
            flags.append(f"INVALID_DATA:volume:{d.date()}:negative_volume")
        zero_volume = df[df["volume"] == 0]
        for d in zero_volume.index:
            flags.append(f"SUSPICIOUS_DATA:volume:{d.date()}:zero_volume")

    # Extreme single-day moves: distinguish a real market event (confirmed by
    # a volume spike) from a likely data glitch (no volume corroboration).
    if "adj_close" in df.columns and len(df) >= 21:
        daily_returns = df["adj_close"].pct_change()
        extreme_days = daily_returns[daily_returns.abs() > _EXTREME_DAILY_RETURN_ABS].index
        for d in extreme_days:
            move = daily_returns.loc[d]
            confirmed = False
            if "volume" in df.columns:
                idx = df.index.get_loc(d)
                if idx >= 20:
                    trailing_avg_volume = df["volume"].iloc[idx - 20 : idx].mean()
                    if trailing_avg_volume and df["volume"].iloc[idx] / trailing_avg_volume >= _VOLUME_CONFIRMATION_RATIO:
                        confirmed = True
            label = "VALID_EXTREME_MARKET_MOVE" if confirmed else "SUSPICIOUS_DATA"
            flags.append(f"{label}:returns:{d.date()}:{move:+.1%}")

    return flags


def _validate_indicator_bounds(rsi: Optional[float], atr: Optional[float], volatility_20d: Optional[float], volatility_60d: Optional[float]) -> list[str]:
    """Sanity checks on OUR OWN computed indicators -- these should always
    hold mathematically; a violation points at a bug, not a market condition.
    """
    flags = []
    if rsi is not None and not (0.0 <= rsi <= 100.0):
        flags.append(f"INVALID_DATA:rsi_14:{rsi:.2f}:out_of_range")
    for name, value in (("atr_14", atr), ("volatility_20d", volatility_20d), ("volatility_60d", volatility_60d)):
        if value is not None and value < 0:
            flags.append(f"INVALID_DATA:{name}:{value:.4f}:negative")
    return flags


# ---------------------------------------------------------------------------
# Trend / state classification -- descriptive only, never a recommendation
# ---------------------------------------------------------------------------

def classify_price_trend(price: Optional[float], sma50: Optional[float], sma200: Optional[float]) -> TrendDirection:
    """Price vs SMA50 vs SMA200 relationships (section 5/12). A BULLISH/
    BEARISH label describes price behavior only -- it is NOT a recommendation.
    """
    if price is None or sma50 is None or sma200 is None:
        return "INSUFFICIENT_DATA"
    above_50, above_200, fifty_above_two_hundred = price > sma50, price > sma200, sma50 > sma200
    if above_50 and above_200 and fifty_above_two_hundred:
        return "BULLISH"
    if not above_50 and not above_200 and not fifty_above_two_hundred:
        return "BEARISH"
    return "MIXED"


def classify_momentum(return_20d: Optional[float], return_60d: Optional[float], return_252d: Optional[float]) -> TrendDirection:
    """Direction of the available return horizons -- section 12's "momentum
    trend". Uses whichever horizons have data rather than requiring all three.
    """
    values = [v for v in (return_20d, return_60d, return_252d) if v is not None]
    if not values:
        return "INSUFFICIENT_DATA"
    positive, negative = sum(1 for v in values if v > 0), sum(1 for v in values if v < 0)
    if positive == len(values):
        return "BULLISH"
    if negative == len(values):
        return "BEARISH"
    return "MIXED"


def classify_rsi(rsi: Optional[float]) -> RsiState:
    if rsi is None:
        return "INSUFFICIENT_DATA"
    if rsi >= 70:
        return "OVERBOUGHT"
    if rsi <= 30:
        return "OVERSOLD"
    return "NEUTRAL"


def classify_volatility_regime(current: Optional[float], history: list[TechnicalObservation]) -> VolatilityRegime:
    """LOW/NORMAL/HIGH relative to THIS security's own recent volatility
    distribution (tercile of the trailing history), not a fixed universal
    threshold -- "high volatility" means something different for a utility
    than for a small biotech, and this project has no cross-sectional
    universe yet to derive a fixed threshold from (section 12: "only use
    thresholds that can be justified and documented").
    """
    values = sorted(o.value for o in history if o.value is not None)
    if current is None or len(values) < 10:
        return "INSUFFICIENT_DATA"
    low_cut = values[len(values) // 3]
    high_cut = values[(2 * len(values)) // 3]
    if current <= low_cut:
        return "LOW"
    if current >= high_cut:
        return "HIGH"
    return "NORMAL"


def classify_volume_trend(volume_history: list[TechnicalObservation]) -> VolumeTrend:
    """Reuses `fundamentals.classify_trend`'s exact deterministic delta logic
    (average + latest period-over-period move), relabeled INCREASING/
    DECREASING instead of IMPROVING/DETERIORATING since volume direction has
    no inherent "better/worse" -- it is purely descriptive.
    """
    generic_obs = [
        _GenericObservation(metric="volume", value=o.value, period=o.date, period_type="annual", period_end=o.date, source=o.source, retrieved_at=o.retrieved_at)
        for o in volume_history
    ]
    state = _classify_trend_generic(generic_obs, higher_is_better=True)
    return {"IMPROVING": "INCREASING", "DETERIORATING": "DECREASING", "STABLE": "STABLE", "INSUFFICIENT_DATA": "INSUFFICIENT_DATA"}[state]


# ---------------------------------------------------------------------------
# Output schema (mirrors CompanyFundamentals's shape/spirit)
# ---------------------------------------------------------------------------

class TechnicalAnalysis(BaseModel):
    ticker: str
    as_of: Optional[str] = None
    current_price: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None

    # Returns -- flat scalars for direct consumption (horizon.py/risk_reward.py
    # convention), full provenance in `return_observations`.
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_20d: Optional[float] = None
    return_60d: Optional[float] = None
    return_120d: Optional[float] = None
    return_252d: Optional[float] = None
    return_observations: list[ReturnObservation] = Field(default_factory=list)

    # Moving averages ("ma" naming matches this project's existing
    # price_vs_ma50/price_vs_ma200 fields -- these are simple, unweighted
    # moving averages throughout, never exponential).
    moving_average_20d: Optional[float] = None
    moving_average_50d: Optional[float] = None
    moving_average_100d: Optional[float] = None
    moving_average_200d: Optional[float] = None
    price_vs_ma20: Optional[float] = None
    price_vs_ma50: Optional[float] = None  # existing name, unchanged (risk_reward.py/horizon.py depend on it)
    price_vs_ma200: Optional[float] = None  # existing name, unchanged
    ma20_vs_ma50: Optional[float] = None
    ma50_vs_ma200: Optional[float] = None

    # Momentum
    rsi_14d: Optional[float] = None  # existing name, unchanged
    rsi_state: RsiState = "INSUFFICIENT_DATA"

    # Volatility
    volatility_1m: Optional[float] = None  # existing name (~21d window), unchanged -- strategy.py/horizon.py depend on it
    volatility_20d: Optional[float] = None  # this brief's exact requested window (20d)
    volatility_60d: Optional[float] = None
    atr_14d: Optional[float] = None  # existing name, unchanged
    atr_14d_pct: Optional[float] = None  # ATR as % of current price
    volatility_regime: VolatilityRegime = "INSUFFICIENT_DATA"

    # Volume
    volume_ratio: Optional[float] = None  # existing name (~21d window), unchanged -- risk_reward.py/horizon.py depend on it
    average_volume_20d: Optional[float] = None
    volume_vs_average_20d: Optional[float] = None
    volume_trend: VolumeTrend = "INSUFFICIENT_DATA"

    # Relative strength (section 9)
    benchmark_ticker: Optional[str] = None
    relative_return_20d: Optional[float] = None
    relative_return_60d: Optional[float] = None
    relative_return_252d: Optional[float] = None
    benchmark_return_20d: Optional[float] = None
    benchmark_return_60d: Optional[float] = None
    benchmark_return_252d: Optional[float] = None

    # Support/resistance -- same swing-high/low methodology as support_60d/
    # resistance_60d (kept for risk_reward.py backward compatibility);
    # nearest_support/nearest_resistance/distance_*_pct are this brief's
    # requested names for the identical computation, not a second method.
    support_60d: Optional[float] = None
    resistance_60d: Optional[float] = None
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    distance_to_support_pct: Optional[float] = None
    distance_to_resistance_pct: Optional[float] = None

    # Trend classification -- descriptive, never a recommendation
    trend: TrendDirection = "INSUFFICIENT_DATA"
    momentum_state: TrendDirection = "INSUFFICIENT_DATA"

    # Freshness / provenance
    data_source: str = DATA_SOURCE
    retrieved_at: Optional[str] = None
    history_start: Optional[str] = None
    history_end: Optional[str] = None
    data_quality: list[str] = Field(default_factory=list)

    # Full historical indicator series (section 11)
    history: dict[str, list[TechnicalObservation]] = Field(default_factory=dict)


def _latest(observations: list[TechnicalObservation]) -> Optional[float]:
    for obs in reversed(observations):
        if obs.value is not None:
            return obs.value
    return None


def _compute_returns(prices: pd.DataFrame) -> tuple[dict[str, Optional[float]], list[ReturnObservation]]:
    flat: dict[str, Optional[float]] = {}
    observations: list[ReturnObservation] = []
    sorted_prices = prices.sort_index()

    for label, window in RETURN_WINDOWS.items():
        value = market_features.compute_returns(prices, window=window)
        complete = len(sorted_prices) >= window + 1
        start_date = end_date = None
        if complete:
            end_date = str(sorted_prices.index[-1].date())
            start_date = str(sorted_prices.index[-1 - window].date())
        flat[f"return_{label}"] = value
        observations.append(ReturnObservation(horizon=label, start_date=start_date, end_date=end_date, value=value, complete=complete))

    return flat, observations


def compute_technicals(
    ticker: str,
    price_prov: Optional[price_provider.PriceProvider] = None,
    benchmark_ticker: str = config.DEFAULT_BENCHMARK_TICKER,
    benchmark_prov: Optional[price_provider.PriceProvider] = None,
) -> dict:
    """Build the full `TechnicalAnalysis` object for one ticker, as a plain dict.

    Never raises on a provider failure -- one ticker's technicals failing
    (bad ticker, network issue) must not crash a batch/UI render, same
    principle as `fundamentals.compute_fundamentals`.
    """
    ticker = ticker.upper()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    try:
        prices = price_provider.get_price_history(ticker, provider=price_prov)
    except Exception:
        logger.exception("Price history fetch failed for %s -- returning an empty technical snapshot", ticker)
        return TechnicalAnalysis(ticker=ticker, retrieved_at=retrieved_at, data_quality=["INVALID_DATA:provider:fetch_failed"]).model_dump()

    if prices.empty:
        return TechnicalAnalysis(ticker=ticker, retrieved_at=retrieved_at, data_quality=["INSUFFICIENT_HISTORY:no_price_data"]).model_dump()

    prices = prices.sort_index()
    data_quality = validate_ohlcv(prices)

    latest_row = prices.iloc[-1]
    as_of = str(prices.index[-1].date())
    current_price = float(latest_row["adj_close"]) if pd.notna(latest_row.get("adj_close")) else None

    returns_flat, return_observations = _compute_returns(prices)

    ma20 = market_features.compute_moving_average(prices, window=20)
    ma50 = market_features.compute_moving_average(prices, window=50)
    ma100 = market_features.compute_moving_average(prices, window=100)
    ma200 = market_features.compute_moving_average(prices, window=200)

    rsi14 = market_features.compute_rsi(prices, window=14)
    atr14 = market_features.compute_atr(prices, window=14)
    vol20 = market_features.compute_volatility(prices, window=20)
    vol60 = market_features.compute_volatility(prices, window=60)
    vol1m = market_features.compute_volatility(prices, window=21)

    data_quality += _validate_indicator_bounds(rsi14, atr14, vol20, vol60)

    swing = market_features.compute_swing_levels(prices, window=60)
    volume_ratio = market_features.compute_volume_ratio(prices, window=21)
    average_volume_20d = float(prices["volume"].tail(20).mean()) if "volume" in prices.columns and len(prices) >= 20 else None
    volume_vs_average_20d = (
        float(prices["volume"].iloc[-1] / average_volume_20d) if (average_volume_20d and "volume" in prices.columns) else None
    )

    history = {
        "close": _series_over_history(prices, "close", lambda p, as_of=None: (p.loc[p.index <= as_of, "adj_close"].iloc[-1] if not p.loc[p.index <= as_of].empty else None), retrieved_at),
        "sma_20": _series_over_history(prices, "sma_20", market_features.compute_moving_average, retrieved_at, window=20),
        "sma_50": _series_over_history(prices, "sma_50", market_features.compute_moving_average, retrieved_at, window=50),
        "rsi_14": _series_over_history(prices, "rsi_14", market_features.compute_rsi, retrieved_at, window=14),
        "atr_14": _series_over_history(prices, "atr_14", market_features.compute_atr, retrieved_at, window=14),
        "volatility_20d": _series_over_history(prices, "volatility_20d", market_features.compute_volatility, retrieved_at, window=20),
        "volume": _series_over_history(prices, "volume", lambda p, as_of=None: (float(p.loc[p.index <= as_of, "volume"].iloc[-1]) if ("volume" in p.columns and not p.loc[p.index <= as_of].empty) else None), retrieved_at),
    }

    # Relative strength vs benchmark -- missing benchmark data means missing
    # relative-strength fields, never a silently-skipped or zero-filled value.
    benchmark_returns: dict[str, Optional[float]] = {"20d": None, "60d": None, "252d": None}
    try:
        benchmark_prices = price_provider.get_price_history(benchmark_ticker, provider=benchmark_prov)
        for label in ("20d", "60d", "252d"):
            benchmark_returns[label] = market_features.compute_returns(benchmark_prices, window=RETURN_WINDOWS[label])
    except Exception:
        logger.warning("Benchmark (%s) price history unavailable -- relative strength will be missing, not fabricated", benchmark_ticker)

    def _relative(label: str) -> Optional[float]:
        stock_return, bench_return = returns_flat.get(f"return_{label}"), benchmark_returns.get(label)
        return (stock_return - bench_return) if (stock_return is not None and bench_return is not None) else None

    distance_to_support_pct = (
        (current_price / swing["support"] - 1) if (current_price is not None and swing["support"]) else None
    )
    distance_to_resistance_pct = (
        (swing["resistance"] / current_price - 1) if (current_price is not None and swing["resistance"]) else None
    )

    result = TechnicalAnalysis(
        ticker=ticker,
        as_of=as_of,
        current_price=current_price,
        open=float(latest_row["open"]) if "open" in prices.columns and pd.notna(latest_row.get("open")) else None,
        high=float(latest_row["high"]) if "high" in prices.columns and pd.notna(latest_row.get("high")) else None,
        low=float(latest_row["low"]) if "low" in prices.columns and pd.notna(latest_row.get("low")) else None,
        volume=float(latest_row["volume"]) if "volume" in prices.columns and pd.notna(latest_row.get("volume")) else None,
        return_1d=returns_flat["return_1d"], return_5d=returns_flat["return_5d"],
        return_20d=returns_flat["return_20d"], return_60d=returns_flat["return_60d"],
        return_120d=returns_flat["return_120d"], return_252d=returns_flat["return_252d"],
        return_observations=return_observations,
        moving_average_20d=ma20, moving_average_50d=ma50, moving_average_100d=ma100, moving_average_200d=ma200,
        price_vs_ma20=market_features.compute_price_vs_ma(prices, window=20),
        price_vs_ma50=market_features.compute_price_vs_ma(prices, window=50),
        price_vs_ma200=market_features.compute_price_vs_ma(prices, window=200),
        ma20_vs_ma50=(ma20 / ma50 - 1) if (ma20 is not None and ma50) else None,
        ma50_vs_ma200=(ma50 / ma200 - 1) if (ma50 is not None and ma200) else None,
        rsi_14d=rsi14, rsi_state=classify_rsi(rsi14),
        volatility_1m=vol1m, volatility_20d=vol20, volatility_60d=vol60,
        atr_14d=atr14, atr_14d_pct=(atr14 / current_price) if (atr14 is not None and current_price) else None,
        volatility_regime=classify_volatility_regime(vol20, history["volatility_20d"]),
        volume_ratio=volume_ratio, average_volume_20d=average_volume_20d, volume_vs_average_20d=volume_vs_average_20d,
        volume_trend=classify_volume_trend(history["volume"]),
        benchmark_ticker=benchmark_ticker,
        relative_return_20d=_relative("20d"), relative_return_60d=_relative("60d"), relative_return_252d=_relative("252d"),
        benchmark_return_20d=benchmark_returns["20d"], benchmark_return_60d=benchmark_returns["60d"], benchmark_return_252d=benchmark_returns["252d"],
        support_60d=swing["support"], resistance_60d=swing["resistance"],
        nearest_support=swing["support"], nearest_resistance=swing["resistance"],
        distance_to_support_pct=distance_to_support_pct, distance_to_resistance_pct=distance_to_resistance_pct,
        trend=classify_price_trend(current_price, ma50, ma200),
        momentum_state=classify_momentum(returns_flat["return_20d"], returns_flat["return_60d"], returns_flat["return_252d"]),
        data_source=DATA_SOURCE,
        retrieved_at=retrieved_at,
        history_start=str(prices.index[0].date()),
        history_end=as_of,
        data_quality=data_quality,
        history=history,
    )
    return result.model_dump()
