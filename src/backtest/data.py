"""Data access for the backtest -- its own caches, never the app's.

The app keeps 2 years of prices (enough for a 252-day lookback). A backtest
needs a warm-up year plus as many evaluation years as possible, so prices go
to `data/raw/prices_long/` with a longer window. Fundamentals reuse the app's
cached raw yfinance statements unchanged (they carry every period yfinance
returns anyway).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config
import price_provider

logger = logging.getLogger(__name__)

LONG_PRICES_DIR = config.RAW_DATA_DIR / "prices_long"
BACKTEST_DIR = config.PROCESSED_DATA_DIR / "backtest"
# Point-in-time 13F universe (STEP 11b) -- backtest-only, never read by the app.
UNIVERSE_HISTORY_PATH = BACKTEST_DIR / "universe_history.parquet"
DEFAULT_YEARS = 10


def _path(ticker: str) -> Path:
    return LONG_PRICES_DIR / f"{ticker.upper()}.parquet"


def load_long_prices(ticker: str, years: int = DEFAULT_YEARS, refresh: bool = False) -> pd.DataFrame:
    """Cached long price history; raises ValueError if nothing is available."""
    path = _path(ticker)
    if path.exists() and not refresh and price_provider._cache_is_fresh(path):
        return pd.read_parquet(path)
    prices = price_provider._download(ticker.upper(), years=years)
    if prices is None:
        if path.exists():
            logger.warning("Download failed for %s; using stale long-history cache", ticker)
            return pd.read_parquet(path)
        raise ValueError(f"No price data for {ticker}")
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


EXT_PRICES_DIR = config.RAW_DATA_DIR / "prices_ext"
EXT_YEARS = 14  # PHASE 1: entries from 2013 need 2012 prices for the 52-week high


def load_ext_prices(ticker: str, years: int = EXT_YEARS, refresh: bool = False) -> pd.DataFrame:
    """PHASE 1 price history (own cache, never the app's): adjusted OHLC plus what market cap needs.

    Columns: adj_close/open/high/low (split+dividend adjusted, as everywhere else),
    volume, `close_split_adj` (yfinance's Close: split-adjusted, NOT dividend-adjusted)
    and `split` (split ratio on its ex-date, 0 otherwise). The price actually
    quoted on a past date = close_split_adj x product of later splits
    (`raw_close`) -- needed because XBRL share counts are as reported, not split-adjusted.

    Frozen once downloaded (no expiry): a research run must not change under it.
    """
    path = EXT_PRICES_DIR / f"{ticker.upper()}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    import yfinance as yf

    raw = yf.download(ticker.upper(), period=f"{years}y", interval="1d", auto_adjust=False, actions=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise ValueError(f"No price data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna(subset=["Close", "Adj Close"])
    ratio = raw["Adj Close"] / raw["Close"]
    prices = pd.DataFrame({
        "adj_close": raw["Adj Close"],
        "open": raw["Open"] * ratio,
        "high": raw["High"] * ratio,
        "low": raw["Low"] * ratio,
        "volume": raw["Volume"],
        "close_split_adj": raw["Close"],
        "split": raw["Stock Splits"] if "Stock Splits" in raw.columns else 0.0,
    })
    prices.index.name = "date"
    if prices.empty:
        raise ValueError(f"No price data for {ticker}")
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


def raw_close(prices: pd.DataFrame) -> pd.Series:
    """The close as quoted that day: undo later splits (yfinance back-adjusts every split)."""
    splits = prices["split"].where(prices["split"] > 0, 1.0)
    later = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)  # product of splits strictly after each day
    return prices["close_split_adj"] * later


def load_raw_fundamentals(ticker: str) -> dict:
    import fundamentals

    return fundamentals.YFinanceFundamentalsProvider().get_raw_fundamentals(ticker)


def load_universe_history() -> pd.DataFrame:
    return pd.read_parquet(UNIVERSE_HISTORY_PATH)


def universe_tickers() -> list[str]:
    """Tickers in the current institutional universe (survivorship-biased: see README)."""
    path = config.INSTITUTIONAL_UNIVERSE_PATH
    if not path.exists():
        return []
    universe = pd.read_parquet(path)
    if "ticker" not in universe.columns:
        return []
    return sorted({str(t).upper() for t in universe["ticker"].dropna() if str(t).strip()})
