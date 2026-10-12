"""10-K / 10-Q Risk Factors and MD&A text, point-in-time, for "Lazy Prices" (PHASE 1b).

Cohen, Malloy & Nguyen (2020), "Lazy Prices": when a company substantially
rewrites its annual/quarterly report versus the same report a year earlier,
bad news tends to follow -- and the market is slow to read it. The signal is
simply how similar this year's text is to last year's.

This module only produces the text; `lazy_prices.py` turns it into features.

- Filing list: each issuer's SEC submissions JSON (recent + older pages),
  original 10-K and 10-Q only, with the SEC FILING date (the date the text
  became public -- never the period end).
- Text: the primary document, HTML -> text with `fetch_edgar.html_to_text`,
  then the Risk Factors (Item 1A) and MD&A (Item 7 in a 10-K, Item 2 in a
  10-Q) sections. Table-of-contents entries are skipped the same way
  `process_documents.py` does; when a heading appears several times, the
  occurrence with the longest body wins.
- Cache: one parquet per CIK under data/raw/edgar_sections/, resumable.
  (One file per filing would be ~40k files in a OneDrive-synced folder.)
- SEC fair access: one shared rate limiter across worker threads,
  <= RATE_PER_SECOND requests/s, the project's User-Agent.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

SECTIONS_DIR = config.RAW_DATA_DIR / "edgar_sections"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
HEADERS = {"User-Agent": "quant-research-lab research antonin.brengetto@gmail.com", "Accept-Encoding": "gzip, deflate"}
RATE_PER_SECOND = 7.5
FORMS = ("10-K", "10-Q")
MIN_SECTION_CHARS = 1500  # shorter = "see exhibit 13" / "no material changes" stubs, not a section
MAX_SECTION_CHARS = 400_000  # longer = the end boundary was missed (e.g. a combined annual report), not a section
COLUMNS = ["cik", "accession", "form", "filing_date", "report_date", "risk_factors", "mdna", "status"]

ITEM_LINE = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\b\s*[\.\:\-–—]?\s*(.*)$", re.IGNORECASE)
RISK_TITLE = re.compile(r"^risk\s*factors\.?$", re.IGNORECASE)
MDNA_TITLE = re.compile(r"^management['’`]?s\s*discussion\s*and\s*analysis", re.IGNORECASE)
SECTION_ITEMS = {("risk_factors", "10-K"): "1a", ("risk_factors", "10-Q"): "1a",
                 ("mdna", "10-K"): "7", ("mdna", "10-Q"): "2"}
SECTION_TITLES = {"risk_factors": RISK_TITLE, "mdna": MDNA_TITLE}


# ----------------------------------------------------------------------------
# Rate-limited HTTP shared by all threads
# ----------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_time)
            self.next_time = slot + self.interval
        time.sleep(max(0.0, slot - time.monotonic()))


_limiter = RateLimiter(RATE_PER_SECOND)
_session = requests.Session()


def sec_get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        _limiter.wait()
        try:
            response = _session.get(url, headers=HEADERS, timeout=60)
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"SEC request failed after {retries} tries: {url}")


# ----------------------------------------------------------------------------
# Filing list
# ----------------------------------------------------------------------------


def _block_filings(block: dict) -> list[dict]:
    out = []
    for i, form in enumerate(block.get("form", [])):
        if form in FORMS:
            out.append({"accession": block["accessionNumber"][i], "form": form,
                        "filing_date": block["filingDate"][i], "report_date": block.get("reportDate", [None] * (i + 1))[i],
                        "primary_document": block["primaryDocument"][i]})
    return out


def list_filings(cik: int, since: str, fetch: Callable[[str], bytes] = sec_get) -> list[dict]:
    """Original 10-K / 10-Q filings of one issuer filed on/after `since`, oldest first."""
    data = json.loads(fetch(SUBMISSIONS_URL.format(cik=int(cik))))
    filings = _block_filings(data.get("filings", {}).get("recent", {}))
    for page in data.get("filings", {}).get("files", []):
        if page.get("filingTo") and page["filingTo"] < since:
            continue
        filings += _block_filings(json.loads(fetch(SUBMISSIONS_PAGE_URL.format(name=page["name"]))))
    filings = [f for f in filings if f["filing_date"] >= since and f["primary_document"]]
    return sorted({f["accession"]: f for f in filings}.values(), key=lambda f: f["filing_date"])


# ----------------------------------------------------------------------------
# Section extraction
# ----------------------------------------------------------------------------


def _toc_like(lines: list[str], idx: int) -> bool:
    from process_documents import _looks_like_toc_entry

    return _looks_like_toc_entry(lines, idx)


def _next_nonblank(lines: list[str], i: int) -> tuple[int, str]:
    for j in range(i + 1, min(i + 6, len(lines))):
        if lines[j].strip():
            return j, lines[j].strip()
    return i, ""


def _item_of(line: str) -> tuple[Optional[str], str]:
    """('1a', rest-of-line) for an "Item 1A. Risk Factors" line, (None, line) otherwise."""
    m = ITEM_LINE.match(line) if len(line) <= 200 else None
    return (m.group(1).lower(), m.group(2).strip()) if m else (None, line)


def _headings(lines: list[str], item: str, title: re.Pattern) -> list[int]:
    """Line indices where the section's body starts, for any of:
    "Item 1A. Risk Factors" on one line; "Item 1A." then "Risk Factors" on the next
    line (older filings); a bare "Risk Factors" title line. TOC entries are skipped."""
    starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue
        found, rest = _item_of(stripped)
        if found == item:
            if rest and title.match(rest):
                head = i
            elif not rest:
                j, nxt = _next_nonblank(lines, i)
                if not title.match(nxt):
                    continue
                head = j
            else:
                continue
        elif found is None and title.match(stripped):
            head = i
        else:
            continue
        if not _toc_like(lines, head):
            starts.append(head)
    return starts


def _section(lines: list[str], item: str, title: re.Pattern) -> Optional[str]:
    """Body from the heading to the next heading of a DIFFERENT item (page headers
    repeating the same "Item 1A" are skipped); the longest candidate wins."""
    best = None
    for head in _headings(lines, item, title):
        end = len(lines)
        for j in range(head + 1, len(lines)):
            found, _ = _item_of(lines[j].strip())
            if found is not None and found != item:
                end = j
                break
        body = "\n".join(lines[head + 1:end]).strip()
        if best is None or len(body) > len(best):
            best = body
    return best if best and MIN_SECTION_CHARS <= len(best) <= MAX_SECTION_CHARS else None


def extract_sections(text: str, form: str) -> dict:
    lines = text.splitlines()
    return {name: _section(lines, SECTION_ITEMS[(name, form)], SECTION_TITLES[name]) for name in ("risk_factors", "mdna")}


def decode(raw: bytes) -> str:
    """UTF-8, else Windows-1252 (older EDGAR HTML), never silently mangled."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


# ----------------------------------------------------------------------------
# Per-issuer download (resumable)
# ----------------------------------------------------------------------------


def _cache_path(cik: int) -> Path:
    return SECTIONS_DIR / f"{int(cik)}.parquet"


def load_sections(cik: int) -> pd.DataFrame:
    path = _cache_path(cik)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)


def fetch_issuer(cik: int, since: str, fetch: Callable[[str], bytes] = sec_get) -> dict:
    """Download and extract every missing 10-K/10-Q of one issuer; returns a status summary."""
    from fetch_edgar import html_to_text

    have = load_sections(cik)
    done = set(have["accession"])
    try:
        filings = [f for f in list_filings(cik, since, fetch) if f["accession"] not in done]
    except Exception as exc:
        return {"cik": cik, "status": f"listing failed: {exc}", "new": 0}
    rows = []
    for f in filings:
        url = DOC_URL.format(cik=int(cik), acc=f["accession"].replace("-", ""), doc=f["primary_document"])
        try:
            raw = decode(fetch(url))
            text = html_to_text(raw) if "<" in raw[:2000] else raw
            sections = extract_sections(text, f["form"])
            status = "ok" if any(sections.values()) else "no_sections"
        except Exception as exc:
            sections, status = {"risk_factors": None, "mdna": None}, f"error: {type(exc).__name__}"
        rows.append({"cik": int(cik), "accession": f["accession"], "form": f["form"], "filing_date": f["filing_date"],
                     "report_date": f["report_date"], **sections, "status": status})
    if rows:
        frame = pd.concat([have, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
        path = _cache_path(cik)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="zstd")
    return {"cik": cik, "status": "ok", "new": len(rows)}


def _init_worker(per_second: float) -> None:
    """Each worker process gets its share of the SEC request budget (the processes don't share a lock)."""
    global _limiter
    _limiter = RateLimiter(per_second)


def fetch_all(ciks, since: str = "2012-01-01", workers: int = 12, progress: Optional[Callable[[int, int], None]] = None) -> list[dict]:
    """All issuers across worker PROCESSES, one issuer per task. One failing issuer never stops the rest.

    Processes, not threads: HTML-to-text is pure Python and CPU-bound (~1 s for
    a large 10-K), so threads serialize on the GIL. The SEC budget
    (RATE_PER_SECOND) is split evenly across the workers.
    """
    from concurrent.futures import ProcessPoolExecutor

    ciks = sorted({int(c) for c in ciks})
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(RATE_PER_SECOND / workers,)) as pool:
        futures = {pool.submit(fetch_issuer, c, since): c for c in ciks}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"cik": futures[future], "status": f"error: {exc}", "new": 0})
            if progress:
                progress(i, len(ciks))
    return results
