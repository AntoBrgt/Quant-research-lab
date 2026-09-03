"""CLI for extracting structured financial signals from SEC chunks or news.

All the actual extraction mechanics (cache lookup, LLM invocation, structured
validation, the hang/runaway guards) live in `signal_extraction.py`. This
script only loads data, filters it, and dispatches to `--dry-run` cost
estimation or a real cache-first extraction run.

Examples:
    python src/extract_signals.py --dry-run
    python src/extract_signals.py --ticker AAPL --max-chunks 10
    python src/extract_signals.py --source news --ticker AAPL
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import cache
import config
import news_provider
import signal_extraction

LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_documents(documents_path: Path = config.DOCUMENTS_PATH) -> pd.DataFrame:
    """Load processed SEC chunks from parquet."""
    if not documents_path.exists():
        raise FileNotFoundError(f"Processed documents file not found: {documents_path}")

    df = pd.read_parquet(documents_path)
    required_columns = {
        "ticker", "filing_type", "filing_date", "section", "chunk_id",
        "chunk_index", "text", "source_file",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"documents.parquet is missing required columns: {sorted(missing)}")

    return df


def load_news(tickers: list[str]) -> pd.DataFrame:
    """Fetch recent headlines and adapt them to the same row shape as SEC chunks."""
    provider = news_provider.YFinanceNewsProvider()
    rows: list[dict] = []

    for ticker in tickers:
        for item in provider.get_headlines(ticker):
            title = item.get("title", "")
            chunk_id = f"{ticker.upper()}_NEWS_{cache.content_hash(title)[:12]}"
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "filing_type": "NEWS",
                    "filing_date": item.get("published_at"),
                    "section": "News",
                    "chunk_id": chunk_id,
                    "chunk_index": 0,
                    "text": title,
                    "source_file": item.get("publisher") or "yfinance",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured financial signals from SEC chunks or news headlines."
    )
    parser.add_argument("--ticker", nargs="+", default=None, help="Optional ticker filter, e.g. --ticker AAPL MSFT JPM")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional cap for testing on a small subset.")
    parser.add_argument("--source", choices=["sec", "news"], default="sec", help="Signal source (default: sec).")
    parser.add_argument("--dry-run", action="store_true", help="Report cost/cache estimate without calling the LLM.")
    return parser.parse_args()


def _load_input(source: str, tickers: Optional[list[str]]) -> pd.DataFrame:
    if source == "news":
        if not tickers:
            raise ValueError("--source news requires at least one --ticker")
        return load_news(tickers)
    return load_documents()


def _output_path(source: str) -> Path:
    return config.NEWS_SIGNALS_PATH if source == "news" else config.SIGNALS_PATH


def main() -> None:
    args = parse_args()
    df = _load_input(args.source, args.ticker)

    if args.dry_run:
        filtered = signal_extraction.filter_items(df, tickers=args.ticker, max_chunks=args.max_chunks)
        estimate = signal_extraction.estimate_run(filtered, source=args.source)
        print(f"Documents/chunks:     {estimate['documents']}")
        print(f"Unique chunks:        {estimate['unique_chunks']}")
        print(f"Already cached:       {estimate['cache_hits']}")
        print(f"Cache misses:         {estimate['cache_misses']}")
        print(f"LLM calls required:   {estimate['llm_calls_required']}")
        print(f"Estimated input size: {estimate['estimated_input_tokens']}")
        return

    signals, summary = signal_extraction.run_extraction(
        df, source=args.source, tickers=args.ticker, max_chunks=args.max_chunks,
    )

    output_path = _output_path(args.source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_path, index=False)
    logger.info("Saved %d signals to %s", len(signals), output_path)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
