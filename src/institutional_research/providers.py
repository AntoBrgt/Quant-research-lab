"""Opt-in downloader for a user-supplied list of institutional report URLs.

This is NOT a crawler and never discovers URLs on its own -- it only fetches
the exact URLs a user lists in `data/raw/institutional_urls.csv`
(institution, report_title, report_type, publication_date, url). This mirrors
`security_master.py`'s pattern exactly: a separate, explicit, network-
dependent step that nothing else calls automatically, gated behind an
opt-in action in the UI, never run as a side effect of loading the universe.

Every fetch:
1. checks that host's `robots.txt` before requesting the actual URL
2. respects a minimum delay between requests to the same host
3. is skipped if the file was already downloaded (no re-fetching on every run)

This is a best-effort convenience for URLs the user has already vetted as
"official public report" (brief section 2's own priority order) -- it is not
a substitute for checking each institution's terms of use.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30
MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST = 2.0
HEADERS = {"User-Agent": "quant-research-lab research (personal, non-commercial) antonin.brengetto@gmail.com"}

URL_LIST_COLUMNS = ["institution", "report_title", "url"]
_CONTENT_TYPE_TO_EXT = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "text/plain": ".txt",
}

_last_request_at: dict[str, float] = {}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80] or "report"


def _robots_allow(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:
        logger.warning("Could not read robots.txt at %s -- proceeding cautiously", robots_url)
        return True  # no robots.txt / unreachable is conventionally treated as allow-all
    return parser.can_fetch(HEADERS["User-Agent"], url)


def _respect_rate_limit(host: str) -> None:
    last = _last_request_at.get(host)
    if last is not None:
        elapsed = time.monotonic() - last
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS_PER_HOST - elapsed)
    _last_request_at[host] = time.monotonic()


def load_url_list(path: Path = config.INSTITUTIONAL_URL_LIST_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=URL_LIST_COLUMNS + ["report_type", "publication_date"])
    df = pd.read_csv(path)
    missing = set(URL_LIST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def fetch_one(row: dict, root: Path = config.INSTITUTIONAL_RAW_DIR) -> dict:
    """Fetch one URL-list row. Returns a result dict; never raises."""
    institution, title, url = row["institution"], row["report_title"], row["url"]
    dest_dir = root / str(institution)
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)

    existing = list(dest_dir.glob(f"{slug}.*"))
    existing = [p for p in existing if not p.name.endswith(".meta.json")]
    if existing:
        return {"institution": institution, "report_title": title, "url": url, "status": "already_downloaded", "path": str(existing[0])}

    if not _robots_allow(url):
        logger.warning("robots.txt disallows fetching %s -- skipped", url)
        return {"institution": institution, "report_title": title, "url": url, "status": "blocked_by_robots_txt", "path": None}

    host = urlparse(url).netloc
    _respect_rate_limit(host)

    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return {"institution": institution, "report_title": title, "url": url, "status": f"error: {exc}", "path": None}

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext = _CONTENT_TYPE_TO_EXT.get(content_type) or (Path(urlparse(url).path).suffix or ".html")
    dest_path = dest_dir / f"{slug}{ext}"
    dest_path.write_bytes(response.content)

    meta = {
        "institution": institution,
        "report_title": title,
        "report_url": url,
        "report_type": row.get("report_type") or None,
        "publication_date": row.get("publication_date") or None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    dest_path.with_suffix(dest_path.suffix + ".meta.json").write_text(
        pd.Series(meta).to_json(), encoding="utf-8"
    )

    return {"institution": institution, "report_title": title, "url": url, "status": "downloaded", "path": str(dest_path)}


def fetch_from_url_list(
    url_list_path: Path = config.INSTITUTIONAL_URL_LIST_PATH,
    root: Path = config.INSTITUTIONAL_RAW_DIR,
) -> list[dict]:
    """Fetch every not-yet-downloaded URL in the list. Explicit, opt-in, network-dependent.

    Never called automatically by `universe.py` or the Streamlit pages'
    default load path -- only by an explicit user action.
    """
    df = load_url_list(url_list_path)
    return [fetch_one(row, root=root) for row in df.to_dict("records")]
