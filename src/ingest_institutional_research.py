"""CLI: ingest institutional research reports -> mentions -> investable universe.

    python src/ingest_institutional_research.py --fetch-urls   # opt-in: download data/raw/institutional_urls.csv
    python src/ingest_institutional_research.py                # parse whatever is under data/raw/institutional/
    python src/ingest_institutional_research.py --max-chunks 20 # cost-controlled test run

Mirrors `process_documents.py` + `extract_signals.py`'s two-stage shape, but
for institutional reports instead of SEC filings: load raw files -> cache-first
LLM extraction -> persist. `--fetch-urls` is the only network step and is
opt-in on purpose (see `institutional_research/providers.py`).
"""

from __future__ import annotations

import argparse
import logging

import config
from institutional_research import documents, parser, providers, universe

LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest institutional research reports into mentions + universe.")
    p.add_argument("--fetch-urls", action="store_true", help="Opt-in: download data/raw/institutional_urls.csv first (network).")
    p.add_argument("--max-chunks", type=int, default=None, help="Cap chunks processed this run (cost control).")
    p.add_argument("--dry-run", action="store_true", help="Report what would be processed without calling the LLM.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.fetch_urls:
        results = providers.fetch_from_url_list()
        for r in results:
            logger.info("%s | %s -> %s", r["institution"], r["report_title"], r["status"])

    loaded_reports = documents.load_all_reports()
    logger.info("Loaded %d report file(s) from %s", len(loaded_reports), config.INSTITUTIONAL_RAW_DIR)

    if not loaded_reports:
        logger.warning(
            "No institutional reports found under %s -- nothing to ingest. "
            "Drop files into data/raw/institutional/<institution>/ or run with --fetch-urls "
            "after populating data/raw/institutional_urls.csv.",
            config.INSTITUTIONAL_RAW_DIR,
        )
        return

    if args.dry_run:
        total_chunks = sum(len(parser.chunk_report(r)) for r in loaded_reports)
        print(f"Reports found:  {len(loaded_reports)}")
        print(f"Total chunks:   {total_chunks}")
        return

    new_mentions, summary = parser.run_extraction(loaded_reports, max_chunks=args.max_chunks)
    combined = parser.save_mentions(new_mentions, config.INSTITUTIONAL_MENTIONS_PATH)
    logger.info("Saved %d new mention(s) (%d total) to %s", len(new_mentions), len(combined), config.INSTITUTIONAL_MENTIONS_PATH)
    logger.info("Summary: %s", summary)

    universe_df = universe.build_universe(combined)
    universe_df.to_parquet(config.INSTITUTIONAL_UNIVERSE_PATH, index=False)
    logger.info("Universe: %d compan(ies) with an identified ticker -> %s", len(universe_df), config.INSTITUTIONAL_UNIVERSE_PATH)


if __name__ == "__main__":
    main()
