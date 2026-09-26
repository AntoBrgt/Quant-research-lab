"""Load raw institutional report files into plain text + report metadata.

Expected layout, mirroring `data/raw/edgar/` exactly:

    data/raw/institutional/<institution>/<report>.{txt,md,pdf,html,htm}
    data/raw/institutional/<institution>/<report>.meta.json   (optional sidecar)

Supported formats today: plain text/markdown, PDF (via `pypdf`), and HTML
(via `fetch_edgar.html_to_text`, reusing the project's existing stdlib-only
HTML-to-text parser rather than adding a new dependency). Rendering a live
webpage is deliberately NOT supported -- the brief lists "official webpages"
as an acceptable source, but fetching and cleanly extracting arbitrary
institution websites is a much larger, fragile problem than parsing a
document a user has already saved (or that `providers.py` downloaded as a
file). A user who wants a webpage ingested saves it as PDF/HTML first, the
same way `data/raw/edgar/` files are pre-fetched rather than scraped live on
every run.

An optional `<report>.meta.json` sidecar
(`{"report_title": ..., "publication_date": "YYYY-MM-DD", "report_url": ...,
"report_type": ...}`) lets a user state the real metadata exactly rather than
relying on filename/content heuristics -- any field present there overrides
the corresponding guess.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from institutional_research.schemas import InstitutionalReport

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}

_DATE_PATTERNS = (
    re.compile(r"(\d{4}-\d{2}-\d{2})"),
    re.compile(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})"),
)


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily so tests never need it installed

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # a single malformed page shouldn't lose the whole document
            logger.warning("Failed to extract text from one page of %s", path.name)
    return "\n\n".join(pages)


def _read_html(path: Path) -> str:
    import fetch_edgar  # local module, reused rather than adding a new HTML parser

    return fetch_edgar.html_to_text(path.read_text(encoding="utf-8", errors="ignore"))


def extract_text(path: Path) -> str:
    """Dispatch to the right extractor by extension. Returns '' on failure, never raises."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _read_pdf(path)
        if suffix in (".html", ".htm"):
            return _read_html(path)
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        logger.exception("Failed to extract text from %s", path)
        return ""


def _guess_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def _guess_publication_date(text: str, filename: str) -> Optional[str]:
    for source in (filename, text[:2000]):
        for pattern in _DATE_PATTERNS:
            match = pattern.search(source)
            if not match:
                continue
            try:
                import pandas as pd

                parsed = pd.to_datetime(match.group(1), errors="coerce")
                if not pd.isna(parsed):
                    return str(parsed.date())
            except Exception:
                continue
    return None


def _load_sidecar(path: Path) -> dict:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.exists():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("Malformed metadata sidecar, ignoring: %s", sidecar)
        return {}


def list_report_files(root: Path = config.INSTITUTIONAL_RAW_DIR) -> list[Path]:
    """All supported report files under data/raw/institutional/<institution>/*."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.endswith(".meta.json")
    )


def load_report(path: Path, root: Path = config.INSTITUTIONAL_RAW_DIR) -> Optional[dict]:
    """Load one report file into {"report": InstitutionalReport, "text": str}.

    `institution` is the immediate parent folder name relative to `root`
    (e.g. `data/raw/institutional/BlackRock/outlook.pdf` -> "BlackRock").
    Returns None (and logs) for an empty/unreadable file rather than raising --
    one bad report should never abort a batch ingestion.
    """
    import hashlib

    text = extract_text(path)
    if not text.strip():
        logger.warning("Skipping empty/unreadable report: %s", path)
        return None

    try:
        institution = path.relative_to(root).parts[0]
    except ValueError:
        institution = path.parent.name

    sidecar = _load_sidecar(path)
    report = InstitutionalReport(
        institution=sidecar.get("institution", institution),
        report_title=sidecar.get("report_title") or _guess_title(path),
        publication_date=sidecar.get("publication_date") or _guess_publication_date(text, path.name),
        report_url=sidecar.get("report_url"),
        report_type=sidecar.get("report_type"),
        source_path=str(path),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        char_count=len(text),
    )
    return {"report": report, "text": text}


def load_all_reports(root: Path = config.INSTITUTIONAL_RAW_DIR) -> list[dict]:
    """Load every supported report file found under `root`. Never raises on a single bad file."""
    loaded = []
    for path in list_report_files(root):
        item = load_report(path, root=root)
        if item is not None:
            loaded.append(item)
    return loaded
