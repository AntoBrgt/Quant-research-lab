"""market_features must never let future price rows change an as-of value.

This is the check that keeps the research/strategy layers honest: a feature
computed "as of" a given date has to be reproducible from only the data that
existed on that date.
"""

import pandas as pd

import market_features


def _prices(n_days: int, start_price: float = 100.0, spike_at_end: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    closes = [start_price + i * 0.1 for i in range(n_days)]
    if spike_at_end:
        closes[-1] = closes[-2] * 3  # a dramatic future spike
    df = pd.DataFrame({"adj_close": closes, "volume": [1_000_000] * n_days}, index=dates)
    df.index.name = "date"
    return df


def test_features_as_of_a_date_are_unchanged_by_a_later_spike():
    baseline = _prices(300, spike_at_end=False)
    with_future_spike = _prices(300, spike_at_end=True)  # extra info added after the cutoff

    as_of = baseline.index[250]

    features_baseline = market_features.compute_features(baseline, as_of=as_of)
    features_with_spike = market_features.compute_features(with_future_spike, as_of=as_of)

    assert features_baseline == features_with_spike


def test_features_as_of_a_date_ignore_rows_appended_after_it():
    prices = _prices(260)
    as_of = prices.index[200]

    features_full_history = market_features.compute_features(prices, as_of=as_of)

    truncated = prices.loc[prices.index <= as_of]
    features_truncated_history = market_features.compute_features(truncated, as_of=as_of)

    assert features_full_history == features_truncated_history


def test_rsi_only_uses_data_up_to_as_of():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    # steadily rising, then a sharp future drop appended after the cutoff
    closes = [100 + i for i in range(20)] + [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    prices = pd.DataFrame({"adj_close": closes, "volume": [1_000_000] * 30}, index=dates)

    as_of = dates[19]
    rsi_before_drop = market_features.compute_rsi(prices, as_of=as_of, window=14)

    # A steady 20-day rise with no down days should read as strongly overbought,
    # not be dragged down by the crash that happens after the cutoff.
    assert rsi_before_drop is not None and rsi_before_drop > 90


def test_moving_average_excludes_future_rows():
    prices = _prices(260, spike_at_end=True)
    as_of = prices.index[-2]  # day before the spike

    ma_before_spike = market_features.compute_moving_average(prices, as_of=as_of, window=50)
    ma_full_history = market_features.compute_moving_average(prices, as_of=None, window=50)

    assert ma_before_spike != ma_full_history
