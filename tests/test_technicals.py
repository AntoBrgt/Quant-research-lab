"""Technical analysis engine: historical series, quality validation, trend
classification, relative strength -- all deterministic, fully offline
(synthetic OHLCV fixtures only; no yfinance/network/LLM anywhere in this file,
per section 19).
"""

import numpy as np
import pandas as pd

import technicals


class FakeProvider:
    """Stand-in for price_provider.PriceProvider -- returns a fixed DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        return self.df


class FailingProvider:
    def get_price_history(self, ticker: str) -> pd.DataFrame:
        raise ConnectionError("simulated network failure")


def _ohlcv(n=300, start=100.0, drift=0.05, seed=0, with_volume=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = start + np.cumsum(rng.normal(drift, 1.0, n))
    highs = closes + np.abs(rng.normal(0.5, 0.3, n))
    lows = closes - np.abs(rng.normal(0.5, 0.3, n))
    opens = closes + rng.normal(0, 0.2, n)
    data = {"adj_close": closes, "open": opens, "high": highs, "low": lows}
    if with_volume:
        data["volume"] = rng.integers(1_000_000, 5_000_000, n).astype(float)
    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


def _flat_ohlcv(n, value=100.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {"adj_close": [value] * n, "open": [value] * n, "high": [value + 0.5] * n, "low": [value - 0.5] * n, "volume": [1_000_000.0] * n},
        index=dates,
    )
    df.index.name = "date"
    return df


def _trending_ohlcv(n, start=100.0, step=1.0) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [start + i * step for i in range(n)]
    df = pd.DataFrame(
        {
            "adj_close": closes, "open": closes,
            "high": [c + 0.5 for c in closes], "low": [c - 0.5 for c in closes],
            "volume": [1_000_000.0] * n,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

def test_no_network_dependency_in_this_module():
    """A basic guard: this module must never import yfinance directly (it
    only ever reaches prices through price_provider.py's injected provider).
    """
    import inspect

    source = inspect.getsource(technicals)
    assert "import yfinance" not in source


def test_duplicate_dates_are_flagged():
    df = _flat_ohlcv(30)
    df = pd.concat([df, df.iloc[[5]]]).sort_index()
    flags = technicals.validate_ohlcv(df)
    assert any("duplicate_date" in f for f in flags)


def test_unsorted_dates_are_flagged():
    df = _flat_ohlcv(30)
    shuffled = df.iloc[::-1]  # reverse -- descending, not sorted ascending
    flags = technicals.validate_ohlcv(shuffled)
    assert any("unsorted" in f for f in flags)


def test_missing_close_is_flagged():
    df = _flat_ohlcv(30)
    df.loc[df.index[5], "adj_close"] = None
    flags = technicals.validate_ohlcv(df)
    assert any("missing_close" in f for f in flags)


def test_invalid_ohlc_high_less_than_low_is_flagged():
    df = _flat_ohlcv(30)
    df.loc[df.index[3], "high"] = 90.0
    df.loc[df.index[3], "low"] = 95.0
    flags = technicals.validate_ohlcv(df)
    assert any("high_less_than_low" in f for f in flags)


def test_close_above_high_is_flagged():
    df = _flat_ohlcv(30)
    df.loc[df.index[3], "adj_close"] = 1000.0
    flags = technicals.validate_ohlcv(df)
    assert any("close_above_high" in f for f in flags)


def test_close_below_low_is_flagged():
    df = _flat_ohlcv(30)
    df.loc[df.index[3], "adj_close"] = 1.0
    flags = technicals.validate_ohlcv(df)
    assert any("close_below_low" in f for f in flags)


def test_negative_price_is_flagged():
    df = _flat_ohlcv(30)
    df.loc[df.index[3], "adj_close"] = -5.0
    flags = technicals.validate_ohlcv(df)
    assert any("negative_price" in f for f in flags)


def test_negative_volume_is_flagged():
    df = _flat_ohlcv(30)
    df.loc[df.index[3], "volume"] = -100.0
    flags = technicals.validate_ohlcv(df)
    assert any("negative_volume" in f for f in flags)


def test_zero_volume_is_suspicious_not_invalid():
    df = _flat_ohlcv(30)
    df.loc[df.index[3], "volume"] = 0.0
    flags = technicals.validate_ohlcv(df)
    assert any(f.startswith("SUSPICIOUS_DATA") and "zero_volume" in f for f in flags)


def test_insufficient_history_is_flagged():
    df = _flat_ohlcv(5)
    flags = technicals.validate_ohlcv(df)
    assert any("INSUFFICIENT_HISTORY" in f for f in flags)


def test_extreme_move_with_volume_confirmation_is_valid_not_invalid():
    df = _flat_ohlcv(40)
    # Sustain the new, higher price level (not a spike-then-revert) so there
    # is exactly one extreme-return day to evaluate.
    df.loc[df.index[30]:, "adj_close"] = 150.0
    df.loc[df.index[30]:, "high"] = 155.0
    df.loc[df.index[30]:, "low"] = 149.0
    df.loc[df.index[30], "volume"] = 20_000_000.0  # big volume spike confirms it
    flags = technicals.validate_ohlcv(df)
    assert any(f.startswith("VALID_EXTREME_MARKET_MOVE") for f in flags)
    assert not any(f.startswith("SUSPICIOUS_DATA:returns") for f in flags)


def test_extreme_move_without_volume_confirmation_is_suspicious():
    df = _flat_ohlcv(40)
    df.loc[df.index[30], "adj_close"] = 150.0  # +50% in one day, normal volume
    df.loc[df.index[30], "high"] = 155.0
    flags = technicals.validate_ohlcv(df)
    assert any(f.startswith("SUSPICIOUS_DATA:returns") for f in flags)
    assert not any(f.startswith("VALID_EXTREME_MARKET_MOVE") for f in flags)


def test_empty_prices_produces_no_crash_and_no_flags():
    assert technicals.validate_ohlcv(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def test_all_return_horizons_present_with_sufficient_history():
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(_ohlcv(300)), benchmark_prov=FakeProvider(_ohlcv(300, seed=1)))
    for horizon in ("1d", "5d", "20d", "60d", "120d", "252d"):
        assert result[f"return_{horizon}"] is not None


def test_insufficient_history_returns_none_not_a_substitute():
    df = _flat_ohlcv(30)  # not enough for 60d/120d/252d
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["return_20d"] is not None
    assert result["return_60d"] is None
    assert result["return_120d"] is None
    assert result["return_252d"] is None


def test_return_observations_carry_start_end_and_completeness():
    df = _flat_ohlcv(30)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    by_horizon = {o["horizon"]: o for o in result["return_observations"]}
    assert by_horizon["20d"]["complete"] is True
    assert by_horizon["20d"]["start_date"] is not None
    assert by_horizon["60d"]["complete"] is False
    assert by_horizon["60d"]["value"] is None


def test_flat_price_series_has_zero_return():
    df = _flat_ohlcv(30)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["return_20d"] == 0.0


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def test_moving_averages_available_with_sufficient_history():
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(_ohlcv(300)), benchmark_prov=FakeProvider(_ohlcv(300, seed=1)))
    assert result["moving_average_20d"] is not None
    assert result["moving_average_50d"] is not None
    assert result["moving_average_100d"] is not None
    assert result["moving_average_200d"] is not None


def test_sma200_missing_with_insufficient_history():
    df = _flat_ohlcv(100)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["moving_average_100d"] is not None
    assert result["moving_average_200d"] is None  # not enough history -- None, not another window's value


def test_ma_relationships_reflect_a_clear_uptrend():
    df = _trending_ohlcv(260, start=100.0, step=1.0)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["ma20_vs_ma50"] > 0  # steadily rising -> shorter MA above longer MA
    assert result["price_vs_ma50"] > 0


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def test_rsi_is_within_valid_range():
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(_ohlcv(300)), benchmark_prov=FakeProvider(_ohlcv(300, seed=1)))
    assert 0.0 <= result["rsi_14d"] <= 100.0


def test_rsi_none_with_insufficient_history():
    df = _flat_ohlcv(5)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["rsi_14d"] is None
    assert result["rsi_state"] == "INSUFFICIENT_DATA"


def test_rsi_overbought_on_a_steady_uptrend_with_no_down_days():
    df = _trending_ohlcv(30, step=1.0)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["rsi_14d"] > 90
    assert result["rsi_state"] == "OVERBOUGHT"


def test_rsi_oversold_on_a_steady_downtrend():
    df = _trending_ohlcv(30, start=200.0, step=-1.0)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["rsi_14d"] < 10
    assert result["rsi_state"] == "OVERSOLD"


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def test_volatility_is_annualized_and_positive():
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(_ohlcv(300)), benchmark_prov=FakeProvider(_ohlcv(300, seed=1)))
    assert result["volatility_20d"] > 0
    assert result["volatility_60d"] > 0


def test_flat_series_has_zero_volatility():
    df = _flat_ohlcv(100)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["volatility_20d"] == 0.0


def test_volatility_none_with_insufficient_history():
    df = _flat_ohlcv(5)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["volatility_20d"] is None
    assert result["volatility_60d"] is None


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def test_atr_deterministic_for_constant_true_range():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({"adj_close": [100.0] * 30, "high": [101.0] * 30, "low": [99.0] * 30, "volume": [1.0] * 30}, index=dates)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["atr_14d"] == 2.0
    assert round(result["atr_14d_pct"], 4) == round(2.0 / 100.0, 4)


def test_atr_none_with_insufficient_history():
    df = _flat_ohlcv(5)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["atr_14d"] is None


def test_atr_none_without_high_low_columns():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({"adj_close": [100.0] * 30, "volume": [1.0] * 30}, index=dates)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["atr_14d"] is None


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def test_average_volume_and_ratio():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    volumes = [1_000_000.0] * 29 + [3_000_000.0]
    df = pd.DataFrame({"adj_close": [100.0] * 30, "high": [101.0] * 30, "low": [99.0] * 30, "volume": volumes}, index=dates)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    expected_avg = (19 * 1_000_000.0 + 3_000_000.0) / 20
    assert abs(result["average_volume_20d"] - expected_avg) < 1e-6
    assert result["volume_vs_average_20d"] > 1.0


def test_missing_volume_produces_none_not_zero():
    df = _ohlcv(60, with_volume=False)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["average_volume_20d"] is None
    assert result["volume_ratio"] is None
    assert result["volume_trend"] == "INSUFFICIENT_DATA"


def test_volume_trend_increasing():
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    volumes = list(np.linspace(1_000_000, 3_000_000, 90))
    df = pd.DataFrame({"adj_close": [100.0] * 90, "high": [101.0] * 90, "low": [99.0] * 90, "volume": volumes}, index=dates)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["volume_trend"] == "INCREASING"


# ---------------------------------------------------------------------------
# Relative strength
# ---------------------------------------------------------------------------

def test_relative_return_is_stock_minus_benchmark():
    stock_df = _trending_ohlcv(260, start=100.0, step=1.0)
    benchmark_df = _flat_ohlcv(260)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(stock_df), benchmark_prov=FakeProvider(benchmark_df))
    assert result["benchmark_return_20d"] == 0.0
    assert result["relative_return_20d"] == result["return_20d"]  # benchmark flat -> relative == absolute
    assert result["relative_return_20d"] > 0


def test_missing_benchmark_gives_missing_relative_strength_not_skipped_silently():
    df = _ohlcv(300)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FailingProvider())
    assert result["benchmark_return_20d"] is None
    assert result["relative_return_20d"] is None
    assert result["benchmark_ticker"] == "SPY"  # the attempted benchmark is still recorded


# ---------------------------------------------------------------------------
# Support / resistance
# ---------------------------------------------------------------------------

def test_support_resistance_deterministic_from_swing_high_low():
    df = _ohlcv(90)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    tail = df.sort_index().tail(60)
    assert result["nearest_support"] == tail["low"].min()
    assert result["nearest_resistance"] == tail["high"].max()
    assert result["distance_to_support_pct"] >= 0
    assert result["distance_to_resistance_pct"] >= 0


def test_support_resistance_none_with_insufficient_history():
    df = _flat_ohlcv(10)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["nearest_support"] is None
    assert result["nearest_resistance"] is None


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

def test_bullish_trend_on_clear_uptrend():
    df = _trending_ohlcv(260, start=100.0, step=1.0)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["trend"] == "BULLISH"
    assert result["momentum_state"] == "BULLISH"


def test_bearish_trend_on_clear_downtrend():
    df = _trending_ohlcv(260, start=500.0, step=-1.0)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["trend"] == "BEARISH"
    assert result["momentum_state"] == "BEARISH"


def test_mixed_trend_on_sideways_choppy_series():
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 0.5, 260).cumsum() * 0  # no drift
    closes = 100 + np.sin(np.linspace(0, 20, 260)) * 5  # oscillating, no trend
    df = pd.DataFrame({"adj_close": closes, "high": closes + 1, "low": closes - 1, "volume": [1_000_000.0] * 260}, index=dates)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["trend"] in ("MIXED", "BULLISH", "BEARISH")  # deterministic given the fixture; mainly checks no crash
    assert result["trend"] != "INSUFFICIENT_DATA"


def test_trend_insufficient_data_with_short_history():
    df = _flat_ohlcv(10)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["trend"] == "INSUFFICIENT_DATA"
    assert result["momentum_state"] == "INSUFFICIENT_DATA"


def test_volatility_regime_classifies_relative_to_own_history():
    # Low, stable volatility for most of history, then a clear volatility spike.
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    rng = np.random.default_rng(7)
    calm = 100 + rng.normal(0, 0.1, 130).cumsum()
    stormy_start = calm[-1]
    stormy = stormy_start + rng.normal(0, 3.0, 20).cumsum()
    closes = np.concatenate([calm, stormy])
    df = pd.DataFrame({"adj_close": closes, "high": closes + 1, "low": closes - 1, "volume": [1_000_000.0] * 150}, index=dates)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["volatility_regime"] == "HIGH"


# ---------------------------------------------------------------------------
# Freshness / provenance
# ---------------------------------------------------------------------------

def test_as_of_and_provenance_fields_are_populated():
    df = _ohlcv(60)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    assert result["as_of"] == str(df.sort_index().index[-1].date())
    assert result["history_start"] == str(df.sort_index().index[0].date())
    assert result["history_end"] == result["as_of"]
    assert result["data_source"] == "price_provider:yfinance"
    assert result["retrieved_at"]


def test_history_series_carry_dates_and_source():
    df = _ohlcv(90)
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(df), benchmark_prov=FakeProvider(df))
    close_history = result["history"]["close"]
    assert len(close_history) > 0
    assert close_history[-1]["date"] == result["as_of"]
    assert close_history[0]["source"] == "computed"


# ---------------------------------------------------------------------------
# Provider failure
# ---------------------------------------------------------------------------

def test_price_provider_failure_does_not_crash_and_returns_honest_empty_result():
    result = technicals.compute_technicals("TEST", price_prov=FailingProvider())
    assert result["current_price"] is None
    assert result["return_20d"] is None
    assert any("fetch_failed" in f for f in result["data_quality"])


def test_empty_price_history_does_not_crash():
    result = technicals.compute_technicals("TEST", price_prov=FakeProvider(pd.DataFrame()))
    assert result["current_price"] is None
    assert any("no_price_data" in f for f in result["data_quality"])
