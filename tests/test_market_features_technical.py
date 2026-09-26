"""New technical indicators (ATR, price-vs-MA, swing levels) -- section 7 of the brief.

`test_lookahead_bias.py` already covers the as-of discipline for the original
functions; this file covers correctness of the additions and their graceful
degradation when `high`/`low` aren't available (older cached price files).
"""

import pandas as pd

import market_features


def _ohlc_prices(n_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    closes = [100.0 + i * 0.5 for i in range(n_days)]
    df = pd.DataFrame(
        {
            "adj_close": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "volume": [1_000_000] * n_days,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def test_atr_is_none_without_high_low_columns():
    prices = pd.DataFrame({"adj_close": [100.0 + i for i in range(20)], "volume": [1] * 20})
    assert market_features.compute_atr(prices, window=14) is None


def test_atr_is_positive_with_high_low_columns():
    prices = _ohlc_prices(30)
    atr = market_features.compute_atr(prices, window=14)
    assert atr is not None and atr > 0


def test_atr_matches_hand_computed_value_for_constant_range():
    # A constant 2-point true range (high-low) every day and no gaps between
    # close and next high/low -> ATR should converge to exactly 2.0.
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame({"adj_close": [100.0] * 20, "high": [101.0] * 20, "low": [99.0] * 20, "volume": [1] * 20}, index=dates)
    atr = market_features.compute_atr(df, window=14)
    assert atr == 2.0


def test_price_vs_ma_reflects_direction():
    prices = _ohlc_prices(60)
    above = market_features.compute_price_vs_ma(prices, window=20)
    assert above is not None and above > 0  # prices are rising, so latest > its own trailing MA


def test_price_vs_ma_is_none_with_insufficient_history():
    prices = _ohlc_prices(10)
    assert market_features.compute_price_vs_ma(prices, window=50) is None


def test_swing_levels_use_high_low_when_available():
    prices = _ohlc_prices(60)
    swing = market_features.compute_swing_levels(prices, window=60)
    tail = prices.tail(60)
    assert swing["support"] == tail["low"].min()
    assert swing["resistance"] == tail["high"].max()


def test_swing_levels_fall_back_to_close_only():
    prices = pd.DataFrame({"adj_close": [100.0 + i for i in range(60)], "volume": [1] * 60})
    swing = market_features.compute_swing_levels(prices, window=60)
    assert swing["support"] == prices["adj_close"].min()
    assert swing["resistance"] == prices["adj_close"].max()


def test_compute_features_bundle_includes_new_and_old_keys():
    prices = _ohlc_prices(260)
    features = market_features.compute_features(prices)
    for key in ("return_1d", "return_5d", "return_20d", "return_60d", "return_6m", "return_1y", "atr_14d", "moving_average_20d", "price_vs_ma50", "support_60d", "resistance_60d"):
        assert key in features
    for key in ("return_1m", "return_3m", "moving_average_50d", "moving_average_200d", "rsi_14d", "volume_ratio"):
        assert key in features  # backward compatible with strategy.py's existing weight keys
