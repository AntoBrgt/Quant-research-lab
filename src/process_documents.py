"""Process raw SEC text filings into a chunked, analysis-ready dataset.

This script intentionally operates only on raw EDGAR text files already stored on
local disk. It does not access the SEC network, and it never modifies the raw
source documents.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import cache


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_EDGAR_DIR = PROJECT_ROOT / "data" / "raw" / "edgar"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "documents.parquet"

CHUNK_TARGET_SIZE = 4500
CHUNK_OVERLAP = 200

LOGGER_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)
logger = logging.getLogger(__name__)


SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "Business": (
        r"^\s*item\s*1\.?\s*business\s*$",
        r"^\s*business\s*$",
    ),
    "Risk Factors": (
        r"^\s*item\s*1a\.?\s*risk\s*factors\s*$",
        r"^\s*risk\s*factors\s*$",
    ),
    "Unresolved Staff Comments": (
        r"^\s*item\s*1b\.?\s*unresolved\s*staff\s*comments\s*$",
        r"^\s*unresolved\s*staff\s*comments\s*$",
    ),
    "Properties": (
        r"^\s*item\s*2\.?\s*properties\s*$",
        r"^\s*properties\s*$",
    ),
    "Legal Proceedings": (
        r"^\s*item\s*3\.?\s*legal\s*proceedings\s*$",
        r"^\s*legal\s*proceedings\s*$",
    ),
    "Market for Registrant's Common Equity": (
        r"^\s*item\s*5\.?\s*market\s*for\s*registrant's\s*common\s*equity\s*$",
        r"^\s*market\s*for\s*registrant's\s*common\s*equity\s*$",
    ),
    "Selected Financial Data": (
        r"^\s*item\s*6\.?\s*selected\s*financial\s*data\s*$",
        r"^\s*selected\s*financial\s*data\s*$",
    ),
    "Management's Discussion and Analysis": (
        r"^\s*item\s*7\.?\s*management's\s*discussion\s*and\s*analysis\s*$",
        r"^\s*management's\s+discussion\s+and\s+analysis\s*$",
        r"^\s*md&a\s*$",
    ),
    "Quantitative and Qualitative Disclosures About Market Risk": (
        r"^\s*item\s*7a\.?\s*quantitative\s*and\s*qualitative\s*disclosures\s*about\s*market\s*risk\s*$",
        r"^\s*quantitative\s*and\s*qualitative\s*disclosures\s*about\s*market\s*risk\s*$",
    ),
    "Financial Statements": (
        r"^\s*item\s*8\.?\s*financial\s*statements\s*$",
        r"^\s*financial\s*statements\s*$",
        r"^\s*consolidated\s*financial\s*statements\s*$",
    ),
    "Notes to Financial Statements": (
        r"^\s*notes\s*to\s*financial\s*statements\s*$",
        r"^\s*notes\s*to\s*consolidated\s*financial\s*statements\s*$",
    ),
    "MD&A": (
        r"^\s*management's\s+discussion\s+and\s+analysis\s*$",
        r"^\s*md&a\s*$",
    ),
    "Market Risk": (
        r"^\s*market\s*risk\s*$",
        r"^\s*item\s*7a\.?\s*market\s*risk\s*$",
    ),
    "Full Document": (
        r"^.*$",
    ),
}


def normalize_label(value: str) -> str:
    """Collapse whitespace and punctuation to get a stable search key."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def safe_parse_date(value: Optional[str]) -> Optional[str]:
    """Normalize a date-like string to ISO format when it can be parsed."""
    if value is None:
        return None

    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y-%m-%d")


def parse_filing_type(text: str) -> Optional[str]:
    """Determine whether the filing is a 10-K/10-Q, if clearly stated."""
    match = re.search(r"\b10-Q\b|\b10-K\b", text, flags=re.IGNORECASE)
    if not match:
        return None

    return match.group(0).upper()


def extract_metadata(text: str, source_file: str) -> dict:
    """Extract document metadata when it is present in the filing text.

    The script does not invent information when it cannot be extracted reliably.
    """
    source_name = Path(source_file).name
    ticker = source_name.rsplit(".txt", 1)[0].upper()

    filing_type = parse_filing_type(text)

    filing_date = None
    date_patterns = [
        r"(?i)filed\s+on\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        r"(?i)for\s+the\s+fiscal\s+year\s+ended\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        r"(?i)quarter\s+ended\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        r"(?i)as\s+of\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
        r"\b(\d{4}-\d{2}-\d{2})\b",
    ]

    for pattern in date_patterns:
        m = re.search(pattern, text)
        if not m:
            continue
        filing_date = safe_parse_date(m.group(1))
        if filing_date:
            break

    accession_number = None
    accession_patterns = [
        r"(?i)accession\s+number\s*[:\-]?\s*([0-9]{10}-[0-9]{2}-[0-9]{6})",
        r"\b([0-9]{10}-[0-9]{2}-[0-9]{6})\b",
    ]
    for pattern in accession_patterns:
        m = re.search(pattern, text)
        if m:
            accession_number = m.group(1)
            break

    return {
        "ticker": ticker,
        "filing_type": filing_type,
        "filing_date": filing_date,
        "accession_number": accession_number,
        "source_file": source_name,
    }


def classify_section(line: str) -> Optional[str]:
    """Classify a heading-like SEC line into a canonical section name.

    This is intentionally strict: we only match title-like lines, not ordinary
    descriptive text in the middle of a paragraph. Broad text matches like
    "business" inside a sentence are not valid section headings.
    """
    text = line.strip()
    if not text:
        return None

    # Ignore long narrative lines and body text that happen to contain a keyword.
    if len(text) > 180:
        return None

    normalized = normalize_label(text)
    for section_name, patterns in SECTION_PATTERNS.items():
        if section_name == "Full Document":
            continue

        for pattern in patterns:
            if re.fullmatch(pattern, text, flags=re.IGNORECASE):
                return section_name

    return None


def _is_page_number_line(line: str) -> bool:
    """True for a line that is just a bare page number, e.g. "17"."""
    return bool(re.fullmatch(r"\d{1,4}", line.strip()))


def _looks_like_toc_entry(lines: list[str], header_idx: int, lookahead: int = 10) -> bool:
    """Detect table-of-contents / index rows disguised as section headings.

    In the plain-text renderings we process, a table of contents (or a financial-
    statement index later in the document) lists each heading on its own line and
    is immediately followed a line or two later by a bare page number. A real body
    heading is instead followed by prose. Checking the next non-blank line lets us
    tell the two apart without hardcoding where the table of contents starts or
    ends.
    """
    for line in lines[header_idx + 1 : header_idx + 1 + lookahead]:
        stripped = line.strip()
        if not stripped:
            continue
        return _is_page_number_line(stripped)
    return False


def detect_sections(text: str) -> list[dict]:
    """Split the document into meaningful sections using SEC heading heuristics.

    The plain-text SEC filing is noisy, and formatting varies across documents. The
    strategy is therefore to look for conventional SEC heading names and to capture
    each section as a contiguous block between headings.
    """
    lines = text.splitlines()
    headers: list[tuple[int, str]] = []
    seen_sections: set[str] = set()

    for i, line in enumerate(lines):
        name = classify_section(line)
        if not name:
            continue

        # A single 10-K/10-Q often includes the table of contents (and sometimes a
        # second index further in, e.g. before the financial statements) plus the
        # actual body sections. Table-of-contents rows read as real headings too,
        # so skip anything that looks like one of those index entries.
        if _looks_like_toc_entry(lines, i):
            continue

        # Treating the table of contents and the real body heading as distinct
        # sections would create duplicate chunk IDs for the same logical section.
        # Keep the first real (non-index) occurrence, which is the one closest to
        # the actual document body.
        if name in seen_sections:
            continue

        seen_sections.add(name)
        headers.append((i, name))

    if not headers:
        logger.warning("No recognizable SEC section headings found; using full document")
        return [{"section": "Full Document", "text": text.strip()}]

    sections: list[dict] = []
    for idx, (start_idx, section_name) in enumerate(headers):
        start = start_idx + 1
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)

        content = "\n".join(lines[start:end]).strip()
        if content:
            sections.append({"section": section_name, "text": content})

    if not sections:
        logger.warning("Recognized headings but no non-empty section bodies were found")
        return [{"section": "Full Document", "text": text.strip()}]

    return sections


def find_split_point(text: str, preferred_end: int) -> int:
    """Prefer paragraph or sentence boundaries when chunking long text."""
    search_start = max(0, min(len(text), preferred_end))
    boundary_candidates = [
        text.rfind("\n\n", 0, search_start),
        text.rfind(". ", 0, search_start),
        text.rfind("; ", 0, search_start),
        text.rfind(": ", 0, search_start),
        text.rfind("\n", 0, search_start),
    ]

    valid = [idx for idx in boundary_candidates if idx > 0]
    if not valid:
        return search_start

    return max(valid)


def split_into_chunks(section_text: str, chunk_size: int = CHUNK_TARGET_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a long section into paragraph-friendly chunks with a small overlap."""
    cleaned = re.sub(r"\r\n?", "\n", section_text).strip()
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        candidate = cleaned[start:end].rstrip()

        if end < len(cleaned):
            preferred_end = max(start + chunk_size - overlap, start + 1000)
            split_point = find_split_point(cleaned, min(preferred_end, end))
            if split_point > start + 200:
                end = split_point
                candidate = cleaned[start:end].rstrip()

        if not candidate:
            break

        chunks.append(candidate)

        if end >= len(cleaned):
            break

        next_start = max(start + chunk_size - overlap, end)
        if next_start <= start:
            next_start = start + 1
        start = next_start

    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return cleaned_chunks


def create_chunk_id(ticker: str, filing_type: Optional[str], section: str, index: int) -> str:
    """Create a stable chunk ID for downstream traceability."""
    section_slug = re.sub(r"[^A-Za-z0-9]+", "_", section).strip("_").upper()
    filing_prefix = (filing_type or "DOC").replace("-", "")
    return f"{ticker}_{filing_prefix}_{section_slug}_{index:03d}"


def process_document(file_path: Path) -> list[dict]:
    """Process one raw filing file into section/chunk records."""
    source_name = file_path.name
    logger.info("Processing %s", source_name)

    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not raw_text.strip():
        logger.warning("Skipping empty document: %s", source_name)
        return []

    metadata = extract_metadata(raw_text, source_name)
    sections = detect_sections(raw_text)

    rows: list[dict] = []
    for section in sections:
        section_name = section["section"]
        section_text = section["text"]
        if not section_text.strip():
            continue

        chunks = split_into_chunks(section_text)
        if not chunks:
            logger.warning("No chunks generated for %s / %s", source_name, section_name)
            continue

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = create_chunk_id(
                metadata["ticker"],
                metadata["filing_type"],
                section_name,
                chunk_index,
            )
            rows.append(
                {
                    "ticker": metadata["ticker"],
                    "filing_type": metadata["filing_type"],
                    "filing_date": metadata["filing_date"],
                    "section": section_name,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "content_hash": cache.content_hash(chunk_text),
                    "source_file": metadata["source_file"],
                    "accession_number": metadata["accession_number"],
                    "document_length": len(raw_text),
                }
            )

    logger.info(
        "%s: extracted %d sections and created %d chunks",
        metadata["ticker"],
        len(sections),
        len(rows),
    )

    return rows


def list_documents(tickers: Optional[Iterable[str]] = None) -> list[Path]:
    """List raw EDGAR text files, optionally filtered to selected tickers."""
    if not RAW_EDGAR_DIR.exists():
        raise FileNotFoundError(f"Raw EDGAR directory does not exist: {RAW_EDGAR_DIR}")

    allowed = {ticker.upper() for ticker in (tickers or [])}
    files = sorted(RAW_EDGAR_DIR.glob("*.txt"))

    if allowed:
        files = [
            file_path
            for file_path in files
            if file_path.stem.upper() in allowed
        ]

    return files


def build_dataset(tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Process all raw EDGAR documents and return one chunk-level DataFrame."""
    documents = list_documents(tickers)
    if not documents:
        raise FileNotFoundError(f"No raw EDGAR documents available in {RAW_EDGAR_DIR}")

    rows: list[dict] = []
    for file_path in documents:
        try:
            rows.extend(process_document(file_path))
        except Exception:
            logger.exception("Failed to process %s", file_path.name)
            continue

    if not rows:
        raise ValueError("No valid document chunks were produced from the raw EDGAR inputs")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    expected_columns = [
        "ticker",
        "filing_type",
        "filing_date",
        "section",
        "chunk_id",
        "chunk_index",
        "text",
        "content_hash",
        "source_file",
        "accession_number",
        "document_length",
    ]

    df = df[expected_columns]
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate invariants required for a clean chunk-level document dataset."""
    if df.empty:
        raise ValueError("Processed document dataset is empty.")

    duplicate_mask = df.duplicated(
        subset=["ticker", "filing_type", "section", "chunk_id"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["ticker", "filing_type", "section", "chunk_id", "chunk_index"],
        ]
        raise ValueError(
            "Duplicate chunk IDs detected in processed document dataset:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if not df["chunk_id"].is_unique:
        raise ValueError("chunk_id must be unique across the processed document dataset.")

    # Duplicate content is a real, worth-knowing data-quality signal (e.g. the
    # same boilerplate paragraph chunked twice under different chunk_ids). It is
    # surfaced here rather than silently collapsed with drop_duplicates(), which
    # would just hide it -- the actual fix, if one is needed, belongs in the
    # chunking logic above, not in a downstream filter.
    duplicate_hash_mask = df["content_hash"].duplicated(keep=False)
    if duplicate_hash_mask.any():
        duplicate_groups = df.loc[
            duplicate_hash_mask,
            ["ticker", "section", "chunk_id", "content_hash"],
        ].sort_values("content_hash")
        logger.warning(
            "Detected %d chunks with duplicate content_hash (same content, different chunk_id):\n%s",
            duplicate_hash_mask.sum(),
            duplicate_groups.to_string(index=False),
        )

    logger.info("Dataset invariant check passed: %d rows, %d unique chunk_ids", len(df), df["chunk_id"].nunique())


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a compact diagnostic summary for the generated dataset."""
    print(f"Total rows: {len(df)}")
    print(f"Unique chunk IDs: {df['chunk_id'].nunique()}")
    print(f"Duplicate chunk IDs: {df['chunk_id'].duplicated().sum()}")
    print("Rows per ticker:")
    print(df["ticker"].value_counts().to_string())
    print("Rows per section:")
    print(df["section"].value_counts().to_string())
    print("First 10 chunk IDs:")
    print(df["chunk_id"].head(10).to_list())


def save_dataset(df: pd.DataFrame, output_path: Path = OUTPUT_FILE) -> None:
    """Persist the processed dataset to Parquet after strict validation."""
    validate_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d total chunks to %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw SEC text filings into a chunked document dataset."
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=None,
        help="Optional ticker filters, e.g. --ticker AAPL MSFT JPM",
    )
    args = parser.parse_args()

    tickers = args.ticker
    try:
        df = build_dataset(tickers)
        save_dataset(df)
    except Exception:
        logger.exception("Document processing pipeline failed")
        raise


if __name__ == "__main__":
    main()
