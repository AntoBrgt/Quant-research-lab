import numpy as np
import pandas as pd
import pytest

import live_chart


def _bars(n=260):
    idx = pd.date_range("2026-01-02", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 150, n), index=idx)
    return pd.DataFrame({"Open": close - 0.5, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}, index=idx)


def test_window_follows_horizon():
    assert live_chart.window_for_horizon(3).interval == "5m"
    assert live_chart.window_for_horizon(3).refresh_seconds  # short horizons auto-refresh
    assert live_chart.window_for_horizon(182).interval == "1d"
    assert live_chart.window_for_horizon(182).refresh_seconds is None
    assert live_chart.window_for_horizon(3650).interval == "1wk"


def test_fetch_normalizes_columns_and_handles_empty():
    window = live_chart.window_for_horizon(182)
    bars = live_chart.fetch_bars("X", window, download=lambda t, p, i: _bars())
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert live_chart.fetch_bars("X", window, download=lambda t, p, i: pd.DataFrame()).empty


def test_indicators_match_research_rsi_definition():
    import market_features

    bars = live_chart.fetch_bars("X", live_chart.window_for_horizon(182), download=lambda t, p, i: _bars())
    bars.iloc[-10:, bars.columns.get_loc("close")] -= np.arange(10)  # add some losses
    data = live_chart.add_indicators(bars)
    research_rsi = market_features.compute_rsi(bars.rename(columns={"close": "adj_close"}))
    assert data["rsi14"].iloc[-1] == pytest.approx(research_rsi)
    assert data["sma50"].iloc[-1] == pytest.approx(bars["close"].tail(50).mean())
    assert data["sma200"].notna().any()


def test_short_history_does_not_crash_and_figure_has_all_panels():
    bars = live_chart.fetch_bars("X", live_chart.window_for_horizon(3), download=lambda t, p, i: _bars(30))
    fig = live_chart.build_figure(bars, "X", {"Stop loss": 110.0, "Take profit": None}, intraday=True)
    names = {trace.name for trace in fig.data}
    assert {"X", "SMA20", "Volume", "RSI(14)"} <= names
    assert "SMA200" not in names  # not enough bars -> not drawn, not faked
