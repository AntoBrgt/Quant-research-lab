"""Price data access, behind a small provider interface.

`fetch_prices.py` remains the standalone CLI for bulk-downloading history; this
module is the reusable function other modules (market_features, portfolio,
app) call. Both share the same on-disk cache at `data/raw/prices/{ticker}.parquet`,
so a price already downloaded today is never re-fetched from yfinance within
the same day.

`PriceProvider` is a Protocol rather than an ABC: the codebase only needs a
plain callable shape, and yfinance is the only implementation today. This is
the seam a paid market-data provider would replace later without touching
`market_features.py` or `portfolio.py`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd

import config

logger = logging.getLogger(__name__)

YEARS_OF_HISTORY = 2
# Re-download if the cached file is older than this. Daily price bars don't
# need refreshing more often than once a day for a research prototype.
CACHE_MAX_AGE_HOURS = 20


class PriceProvider(Protocol):
    def get_price_history(self, ticker: str) -> pd.DataFrame:
        """Return a DataFrame indexed by date with an `adj_close` column."""
        ...


def _cache_path(ticker: str) -> Path:
    return config.PRICES_DIR / f"{ticker.upper()}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def _download(ticker: str, years: int = YEARS_OF_HISTORY) -> Optional[pd.DataFrame]:
    """`years` defaults to the app's 2-year window; the backtest
    (`backtest/data.py`) asks for longer history into its own cache."""
    import yfinance as yf  # imported lazily so tests never need it installed

    # auto_adjust=True makes yfinance return a fully split/dividend-adjusted
    # OHLC set (Open/High/Low/Close all adjusted consistently) rather than an
    # adjusted Close alongside raw High/Low. With auto_adjust=False those two
    # would be on different price scales for any ticker with a split or large
    # dividend inside the lookback window -- ATR/support-resistance (from raw
    # High/Low) would then silently disagree with returns/moving averages
    # (from adjusted Close). Every technical indicator in this project uses
    # this single adjusted price series; there is no raw/unadjusted variant
    # kept anywhere, so there is nothing left to accidentally mix.
    data = yf.download(
        ticker,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        logger.error("No price data returned for %s", ticker)
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        logger.error("Close column missing for %s. Columns: %s", ticker, list(data.columns))
        return None

    # `adj_close` is kept as the column name for backward compatibility with
    # every existing caller -- its value is now simply the (already adjusted)
    # Close, not a separately-adjusted series alongside a raw one.
    keep = {"Close": "adj_close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"}
    available = {src: dst for src, dst in keep.items() if src in data.columns}
    prices = data[list(available)].rename(columns=available)
    prices.index.name = "date"
    prices = prices.dropna(subset=["adj_close"])

    return prices if not prices.empty else None


class YFinancePriceProvider:
    """Free, no-API-key price provider backed by yfinance with a disk cache."""

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()
        path = _cache_path(ticker)

        if _cache_is_fresh(path):
            return pd.read_parquet(path)

        prices = _download(ticker)
        if prices is None:
            if path.exists():
                logger.warning("Download failed for %s; falling back to stale cache", ticker)
                return pd.read_parquet(path)
            raise ValueError(f"No price data available for {ticker}")

        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(path)
        return prices


def get_price_history(ticker: str, provider: Optional[PriceProvider] = None) -> pd.DataFrame:
    """Convenience function using the default (yfinance) provider."""
    provider = provider or YFinancePriceProvider()
    return provider.get_price_history(ticker)
