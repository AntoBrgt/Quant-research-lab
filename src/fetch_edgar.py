"""
Fetch the latest SEC 10-K filing for a list of tickers.

If no 10-K is available, the latest 10-Q is fetched instead.

The cleaned filing text is saved to:

    data/raw/edgar/{ticker}.txt

Raw source documents are never overwritten outside of their target file.
"""

import argparse
import html
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# SEC requires a descriptive User-Agent for automated requests.
# Replace the email address with your own.
HEADERS = {
    "User-Agent": "quant-research-lab research antonin.brengetto@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

REQUEST_TIMEOUT = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "edgar"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

class FilingHTMLParser(HTMLParser):
    """
    Very small HTML-to-text parser.

    We deliberately use Python's standard library instead of adding
    BeautifulSoup as a dependency. The goal here is simply to remove
    HTML markup and obvious non-content elements.

    This is sufficient for the prototype. A more sophisticated document
    parser can be introduced later if needed.
    """

    BLOCK_TAGS = {
        "p",
        "div",
        "br",
        "tr",
        "li",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
    }

    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignore_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            self.ignore_depth += 1
            return

        if self.ignore_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            if self.ignore_depth > 0:
                self.ignore_depth -= 1
            return

        if self.ignore_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.ignore_depth == 0:
            self.parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.parts)

        # Decode HTML entities such as &amp; and &nbsp;.
        text = html.unescape(text)

        # Normalize whitespace while keeping paragraph-like line breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()


def html_to_text(html_content: str) -> str:
    """
    Convert SEC filing HTML into reasonably clean plain text.
    """
    parser = FilingHTMLParser()
    parser.feed(html_content)
    parser.close()

    return parser.get_text()


# ---------------------------------------------------------------------------
# SEC helpers
# ---------------------------------------------------------------------------

def get_ticker_to_cik() -> dict[str, str]:
    """
    Download the SEC's official ticker -> CIK mapping.

    CIKs are zero-padded to 10 digits because that is the format expected
    by the SEC submissions endpoint.
    """
    logger.info("Downloading SEC ticker -> CIK mapping")

    response = requests.get(
        SEC_TICKERS_URL,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()

    ticker_to_cik = {}

    for company in data.values():
        ticker = company["ticker"].upper()
        cik = str(company["cik_str"]).zfill(10)
        ticker_to_cik[ticker] = cik

    return ticker_to_cik


def get_latest_filing(
    cik: str,
    preferred_forms: tuple[str, ...] = ("10-K", "10-Q"),
) -> Optional[dict]:
    """
    Return metadata for the latest available 10-K.

    If no 10-K exists, fall back to the latest 10-Q.
    """
    url = SEC_SUBMISSIONS_URL.format(cik=cik)

    logger.info("Fetching SEC submissions for CIK %s", cik)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    recent = data["filings"]["recent"]

    # The SEC submissions endpoint is column-oriented.
    filings = [
        {
            "form": recent["form"][i],
            "filingDate": recent["filingDate"][i],
            "accessionNumber": recent["accessionNumber"][i],
            "primaryDocument": recent["primaryDocument"][i],
        }
        for i in range(len(recent["form"]))
    ]

    for form in preferred_forms:
        matching = [filing for filing in filings if filing["form"] == form]

        if matching:
            # Sort explicitly rather than relying on API ordering.
            matching.sort(
                key=lambda filing: filing["filingDate"],
                reverse=True,
            )

            return matching[0]

    return None


def build_filing_url(cik: str, filing: dict) -> str:
    """
    Build the URL of the primary filing document.
    """
    accession_without_dashes = filing["accessionNumber"].replace("-", "")

    return (
        f"{SEC_ARCHIVES_URL}/"
        f"{int(cik)}/"
        f"{accession_without_dashes}/"
        f"{filing['primaryDocument']}"
    )


def download_filing(url: str) -> str:
    """
    Download a filing document from SEC EDGAR.
    """
    logger.info("Downloading filing: %s", url)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    # SEC filings are generally UTF-8/ASCII-compatible.
    # requests' apparent encoding is used when available.
    response.encoding = response.encoding or "utf-8"

    return response.text


# ---------------------------------------------------------------------------
# Main ticker processing
# ---------------------------------------------------------------------------

def process_ticker(ticker: str, ticker_to_cik: dict[str, str]) -> None:
    """
    Fetch and save the latest filing for one ticker.
    """
    ticker = ticker.upper()

    if ticker not in ticker_to_cik:
        logger.error("Ticker %s not found in SEC ticker mapping", ticker)
        return

    cik = ticker_to_cik[ticker]

    try:
        filing = get_latest_filing(cik)

        if filing is None:
            logger.error("No 10-K or 10-Q found for %s", ticker)
            return

        filing_url = build_filing_url(cik, filing)

        logger.info(
            "%s -> %s filed on %s",
            ticker,
            filing["form"],
            filing["filingDate"],
        )

        raw_html = download_filing(filing_url)
        text = html_to_text(raw_html)

        if not text:
            logger.error("Empty parsed document for %s", ticker)
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = OUTPUT_DIR / f"{ticker}.txt"

        output_path.write_text(
            text,
            encoding="utf-8",
        )

        logger.info(
            "Saved %s (%d characters) -> %s",
            ticker,
            len(text),
            output_path,
        )

    except requests.RequestException as exc:
        logger.error(
            "HTTP error while processing %s: %s",
            ticker,
            exc,
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
        description="Fetch latest SEC 10-K/10-Q filings."
    )

    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbols, e.g. AAPL MSFT JPM",
    )

    args = parser.parse_args()

    ticker_to_cik = get_ticker_to_cik()

    for ticker in args.tickers:
        process_ticker(ticker, ticker_to_cik)


if __name__ == "__main__":
    main()