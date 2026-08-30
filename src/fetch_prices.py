"""
Download adjusted daily prices for a list of tickers.

Data is saved as:

    data/raw/prices/{ticker}.parquet

The script downloads approximately the last two years of daily data.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "prices"

YEARS_OF_HISTORY = 2


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main ticker processing
# ---------------------------------------------------------------------------

def process_ticker(ticker: str) -> None:
    """
    Download and save adjusted daily prices for one ticker.
    """
    ticker = ticker.upper()

    logger.info(
        "Downloading %d years of price data for %s",
        YEARS_OF_HISTORY,
        ticker,
    )

    try:
        data = yf.download(
            ticker,
            period=f"{YEARS_OF_HISTORY}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data.empty:
            logger.error("No price data returned for %s", ticker)
            return

        # yfinance can return a MultiIndex even when downloading one ticker.
        # Flatten it so that the resulting Parquet file stays simple.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Adj Close" not in data.columns:
            logger.error(
                "Adjusted Close column missing for %s. Columns: %s",
                ticker,
                list(data.columns),
            )
            return

        prices = data[["Adj Close"]].copy()

        prices = prices.rename(
            columns={"Adj Close": "adj_close"}
        )

        # Make the index explicit and consistent for downstream processing.
        prices.index.name = "date"

        prices = prices.dropna()

        if prices.empty:
            logger.error("No valid adjusted prices for %s", ticker)
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / f"{ticker}.parquet"

        prices.to_parquet(output_path)

        logger.info(
            "Saved %s rows for %s -> %s",
            len(prices),
            ticker,
            output_path,
        )

    except Exception:
        logger.exception(
            "Unexpected error while processing %s",
            ticker,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download adjusted prices using yfinance."
    )

    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbols, e.g. AAPL MSFT JPM",
    )

    args = parser.parse_args()

    for ticker in args.tickers:
        process_ticker(ticker)


if __name__ == "__main__":
    main()
