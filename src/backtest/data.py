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


def load_raw_fundamentals(ticker: str) -> dict:
    import fundamentals

    return fundamentals.YFinanceFundamentalsProvider().get_raw_fundamentals(ticker)


def universe_tickers() -> list[str]:
    """Tickers in the current institutional universe (survivorship-biased: see README)."""
    path = config.INSTITUTIONAL_UNIVERSE_PATH
    if not path.exists():
        return []
    universe = pd.read_parquet(path)
    if "ticker" not in universe.columns:
        return []
    return sorted({str(t).upper() for t in universe["ticker"].dropna() if str(t).strip()})
