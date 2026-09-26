"""Live price chart for the Company Research page.

Separate from `price_provider.py` on purpose: that provider is a 20h disk
cache of daily bars feeding the (look-ahead-tested) research pipeline. The
chart wants the *latest* bars, including intraday ones for short horizons,
and must never write into the research cache. So this module fetches its
own window straight from yfinance, keeps nothing on disk, and is only ever
used for display.

"Live" here means Yahoo's feed: intraday bars are typically delayed ~15
minutes for US exchanges and nothing updates outside market hours.

The chart window follows the horizon -- you judge a 3-day trade on 5-minute
bars, a 6-month position on a year of daily bars, a 10-year thesis on weekly
bars -- and indicators are computed on whatever bars are shown, with the same
RSI definition as `market_features.compute_rsi` (simple average of the last
14 gains/losses).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class ChartWindow:
    period: str  # yfinance period
    interval: str  # yfinance interval
    description: str
    intraday: bool
    refresh_seconds: Optional[int]  # auto-refresh cadence; None = static


def window_for_horizon(horizon_days: int) -> ChartWindow:
    if horizon_days <= 3:
        return ChartWindow("5d", "5m", "last 5 trading days, 5-minute bars", True, 60)
    if horizon_days <= 14:
        return ChartWindow("1mo", "30m", "last month, 30-minute bars", True, 120)
    if horizon_days <= 91:
        return ChartWindow("6mo", "1d", "last 6 months, daily bars", False, None)
    if horizon_days <= 730:
        return ChartWindow("2y", "1d", "last 2 years, daily bars", False, None)
    return ChartWindow("10y", "1wk", "last 10 years, weekly bars", False, None)


WINDOW_CHOICES: dict[str, ChartWindow] = {
    "1D (1m)": ChartWindow("1d", "1m", "today, 1-minute bars", True, 30),
    "5D (5m)": ChartWindow("5d", "5m", "last 5 trading days, 5-minute bars", True, 60),
    "1M (30m)": ChartWindow("1mo", "30m", "last month, 30-minute bars", True, 120),
    "6M (1d)": ChartWindow("6mo", "1d", "last 6 months, daily bars", False, None),
    "2Y (1d)": ChartWindow("2y", "1d", "last 2 years, daily bars", False, None),
    "10Y (1wk)": ChartWindow("10y", "1wk", "last 10 years, weekly bars", False, None),
}


def _yf_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True, prepost=False)


def fetch_bars(ticker: str, window: ChartWindow, download: Callable[[str, str, str], pd.DataFrame] = _yf_download) -> pd.DataFrame:
    """OHLCV bars (open/high/low/close/volume, DatetimeIndex) for the window. Empty frame if none."""
    raw = download(ticker, window.period, window.interval)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    return bars


def add_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    for window in (20, 50, 200):
        out[f"sma{window}"] = out["close"].rolling(window).mean() if len(out) >= window else float("nan")
    delta = out["close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss
    out["rsi14"] = 100 - 100 / (1 + rs)
    out.loc[avg_loss == 0, "rsi14"] = 100.0
    return out


def build_figure(bars: pd.DataFrame, ticker: str, levels: Optional[dict[str, Optional[float]]] = None, intraday: bool = False):
    """Candles + SMA20/50/200 + key levels / volume / RSI(14), sharing one time axis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = add_indicators(bars)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.62, 0.18, 0.20])

    fig.add_trace(
        go.Candlestick(x=data.index, open=data["open"], high=data["high"], low=data["low"], close=data["close"], name=ticker,
                       increasing_line_color="#1a9e6e", decreasing_line_color="#d0473b"),
        row=1, col=1,
    )
    for column, color in (("sma20", "#3b82c4"), ("sma50", "#e0a526"), ("sma200", "#8a63d2")):
        if data[column].notna().any():
            fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column.upper(), line=dict(width=1.3, color=color)), row=1, col=1)

    level_styles = {
        "Stop loss": ("#d0473b", "dash"), "Take profit": ("#1a9e6e", "dash"),
        "Support": ("#888888", "dot"), "Resistance": ("#888888", "dot"),
    }
    for name, level in (levels or {}).items():
        if level is None or pd.isna(level):
            continue
        color, dash = level_styles.get(name, ("#888888", "dot"))
        fig.add_hline(y=level, line=dict(color=color, dash=dash, width=1), annotation_text=f"{name} {level:,.2f}",
                      annotation_position="top left", row=1, col=1)

    volume_colors = ["#1a9e6e" if c >= o else "#d0473b" for o, c in zip(data["open"], data["close"])]
    fig.add_trace(go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color=volume_colors, opacity=0.6), row=2, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["rsi14"], name="RSI(14)", line=dict(width=1.3, color="#3b82c4")), row=3, col=1)
    for y in (70, 30):
        fig.add_hline(y=y, line=dict(color="#888888", dash="dot", width=1), row=3, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=0), hovermode="x unified")
    if intraday:  # hide nights/weekends so intraday candles don't have huge gaps
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
    else:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig
