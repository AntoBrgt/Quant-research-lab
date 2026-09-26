"""Automatic institutional input: SEC Form 13F-HR holdings.

Why this exists alongside `providers.py`/`parser.py`: outlook reports have to
be found, downloaded, and LLM-parsed by hand, and they mostly talk about
themes rather than companies. A 13F-HR is the opposite -- every US
institutional manager with >$100M in US equities must file its actual
long positions with the SEC each quarter, as structured XML, for free. No
LLM, no scraping, no terms-of-use grey zone.

Pipeline (one call: `refresh_13f_mentions()`):

    SEC submissions JSON (per filer CIK)
        -> latest two 13F-HR filings (current + prior quarter)
        -> information table XML (cached on disk, fetched once per filing)
        -> aggregate rows per CUSIP (shares, value), common stock only
        -> quarter-over-quarter change, flow-adjusted and split-adjusted
        -> select top holdings + biggest movers per filer
        -> CUSIP -> ticker via security_master (OpenFIGI, cached)
        -> InstitutionalMention rows (report_type="13F")

Honesty rules, same as the rest of this project:
- A holding is not a recommendation. `view_direction` describes what the
  filer *did* (added/cut/opened/exited), never "BUY".
- 13Fs are lagged (filed up to 45 days after quarter end), long-only, and US
  equities only. Short positions, non-US holdings, and intra-quarter trades
  are invisible.
- For index-heavy filers (Vanguard/BlackRock/State Street), holdings mostly
  mirror the index and raw share changes mostly mirror fund inflows. That's
  why changes are measured relative to the filer's own median change
  ("flow-adjusted") -- a +3% position in a fund that grew +3% overall is
  flat, not buying.
- Amendments (13F-HR/A) are ignored; the original filing is used.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

SEC_HEADERS = {"User-Agent": "quant-research-lab research antonin.brengetto@gmail.com"}
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"
FILING_FILE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{name}"
REQUEST_TIMEOUT_SECONDS = 60
SEC_MIN_SECONDS_BETWEEN_REQUESTS = 0.15  # SEC fair-access limit is 10 req/s

# Default filers: the five largest 13F filers by reported assets. Override by
# creating data/raw/13f_filers.csv (columns: institution, cik).
DEFAULT_FILERS: dict[str, int] = {
    "BlackRock": 2012383,  # BlackRock, Inc. (post-2024 holding company CIK)
    "Vanguard": 102909,  # VANGUARD GROUP INC
    "State Street": 93751,  # STATE STREET CORP
    "Fidelity": 315066,  # FMR LLC
    "JPMorgan": 19617,  # JPMORGAN CHASE & CO
}

TOP_N_BY_VALUE = 30  # largest positions per filer that enter the universe
TOP_N_MOVERS = 10  # biggest flow-adjusted adds (and, separately, cuts) per filer
MOVER_CANDIDATE_POOL = 300  # movers are only picked among this many largest positions (avoids noise in tiny stakes)
CHANGE_THRESHOLD = 0.10  # |flow-adjusted change| above this is a directional add/cut
EXCLUDED_SECURITY_TYPES = {"ETP", "Open-End Fund", "Closed-End Fund", "Mutual Fund", "Unit"}

_last_request_at = 0.0


# ----------------------------------------------------------------------------
# HTTP (thin, injectable so tests never touch the network)
# ----------------------------------------------------------------------------

Fetcher = Callable[[str], bytes]


def sec_get(url: str) -> bytes:
    global _last_request_at
    wait = SEC_MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    response = requests.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    _last_request_at = time.monotonic()
    response.raise_for_status()
    return response.content


def load_filers(path: Optional[Path] = None) -> dict[str, int]:
    path = path or config.RAW_DATA_DIR / "13f_filers.csv"
    if not path.exists():
        return dict(DEFAULT_FILERS)
    frame = pd.read_csv(path)
    missing = {"institution", "cik"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    return {str(row.institution).strip(): int(row.cik) for row in frame.itertuples()}


# ----------------------------------------------------------------------------
# Filing discovery
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Filing13F:
    cik: int
    accession: str  # dashed form, e.g. 0001067983-26-000012
    filing_date: str
    report_date: str  # quarter end the holdings are as of

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def index_url(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession_nodash}/"


def _filings_from_block(block: dict, cik: int) -> list[Filing13F]:
    forms = block.get("form", [])
    out = []
    for i, form in enumerate(forms):
        if form != "13F-HR":
            continue
        report_date = block.get("reportDate", [None] * len(forms))[i]
        if not report_date:
            continue
        out.append(Filing13F(cik, block["accessionNumber"][i], block["filingDate"][i], report_date))
    return out


def list_13f_filings(cik: int, fetch: Fetcher = sec_get) -> list[Filing13F]:
    """All original 13F-HR filings for a filer, newest quarter first.

    Large filers (e.g. BlackRock files thousands of 13G/Ds) can push their
    13F-HRs out of the `recent` block; older pages under `filings.files` are
    only fetched when `recent` doesn't contain at least two quarters.
    """
    data = json.loads(fetch(SUBMISSIONS_URL.format(cik=cik)))
    filings = _filings_from_block(data.get("filings", {}).get("recent", {}), cik)

    for page in data.get("filings", {}).get("files", []):
        if len({f.report_date for f in filings}) >= 2:
            break
        page_data = json.loads(fetch(SUBMISSIONS_PAGE_URL.format(name=page["name"])))
        filings.extend(_filings_from_block(page_data, cik))

    # One filing per quarter: if a quarter has several originals, keep the latest-filed.
    by_quarter: dict[str, Filing13F] = {}
    for f in filings:
        if f.report_date not in by_quarter or f.filing_date > by_quarter[f.report_date].filing_date:
            by_quarter[f.report_date] = f
    return sorted(by_quarter.values(), key=lambda f: f.report_date, reverse=True)


# ----------------------------------------------------------------------------
# Information table parsing
# ----------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> Optional[str]:
    for child in element.iter():
        if _local(child.tag) == name and child.text is not None:
            return child.text.strip()
    return None


def parse_information_table(xml_bytes: bytes) -> pd.DataFrame:
    """Parse a 13F information table into one row per CUSIP (common/long only).

    - Option rows (`putCall` present) and principal-amount rows (`PRN`, i.e.
      bonds) are dropped: they aren't share positions.
    - A filer often reports one CUSIP on several rows (one per sub-manager /
      discretion type); those are summed.
    - Namespace-agnostic: the SEC schema namespace/prefix varies between filers.
    """
    root = ET.fromstring(xml_bytes)
    rows = []
    for entry in root.iter():
        if _local(entry.tag) != "infoTable":
            continue
        if _child_text(entry, "putCall"):
            continue
        amount_type = (_child_text(entry, "sshPrnamtType") or "SH").upper()
        if amount_type != "SH":
            continue
        cusip = (_child_text(entry, "cusip") or "").upper()
        if not cusip:
            continue
        try:
            shares = float((_child_text(entry, "sshPrnamt") or "0").replace(",", ""))
            value = float((_child_text(entry, "value") or "0").replace(",", ""))
        except ValueError:
            continue
        rows.append(
            {
                "cusip": cusip,
                "name_of_issuer": _child_text(entry, "nameOfIssuer"),
                "title_of_class": _child_text(entry, "titleOfClass"),
                "shares": shares,
                "value": value,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["cusip", "name_of_issuer", "title_of_class", "shares", "value"])

    frame = pd.DataFrame(rows)
    return (
        frame.groupby("cusip", as_index=False)
        .agg(name_of_issuer=("name_of_issuer", "first"), title_of_class=("title_of_class", "first"),
             shares=("shares", "sum"), value=("value", "sum"))
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )


def _cache_path(filing: Filing13F) -> Path:
    return config.RAW_DATA_DIR / "13f" / str(filing.cik) / f"{filing.accession_nodash}.parquet"


def load_holdings(filing: Filing13F, fetch: Fetcher = sec_get) -> pd.DataFrame:
    """Holdings for one filing, from the disk cache when present (a filed 13F never changes)."""
    path = _cache_path(filing)
    if path.exists():
        return pd.read_parquet(path)

    index = json.loads(fetch(FILING_INDEX_URL.format(cik=filing.cik, acc=filing.accession_nodash)))
    items = index.get("directory", {}).get("item", [])
    xml_names = [i["name"] for i in items if i["name"].lower().endswith(".xml") and i["name"].lower() != "primary_doc.xml"]

    frames = []
    for name in xml_names:
        content = fetch(FILING_FILE_URL.format(cik=filing.cik, acc=filing.accession_nodash, name=name))
        try:
            parsed = parse_information_table(content)
        except ET.ParseError:
            logger.warning("Could not parse %s in %s", name, filing.accession)
            continue
        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        raise ValueError(f"No information table found in 13F filing {filing.accession}")

    holdings = pd.concat(frames, ignore_index=True)
    if len(frames) > 1:  # information table split across files
        holdings = (
            holdings.groupby("cusip", as_index=False)
            .agg(name_of_issuer=("name_of_issuer", "first"), title_of_class=("title_of_class", "first"),
                 shares=("shares", "sum"), value=("value", "sum"))
            .sort_values("value", ascending=False)
            .reset_index(drop=True)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    holdings.to_parquet(path, index=False)
    return holdings


# ----------------------------------------------------------------------------
# Quarter-over-quarter change
# ----------------------------------------------------------------------------


def _split_factor(share_ratio: float, price_ratio: float) -> float:
    """Detect a stock split from the filing-implied price (value/shares).

    A 4:1 split shows up as shares x4 and implied price /4 in the same
    quarter. Without this, every split would look like massive buying.
    Returns the factor to divide the share ratio by (1.0 = no split).
    """
    if not (price_ratio > 0 and share_ratio > 0):
        return 1.0
    forward = 1.0 / price_ratio  # e.g. 4.0 after a 4:1 split
    k = round(forward)
    if k >= 2 and abs(forward - k) / k < 0.15 and 0.6 < share_ratio / k < 1.6:
        return float(k)
    k = round(price_ratio)  # reverse split: price x k, shares / k
    if k >= 2 and abs(price_ratio - k) / k < 0.15 and 0.6 < share_ratio * k < 1.6:
        return 1.0 / k
    return 1.0


def compare_quarters(current: pd.DataFrame, previous: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Current holdings + weight + flow- and split-adjusted change vs the prior quarter.

    Adds columns: weight, prev_shares, share_change (raw), adjusted_change
    (relative to the filer's median share change, split-corrected), status
    (NEW / ADDED / REDUCED / UNCHANGED / EXITED / NO_PRIOR).
    """
    current = current.copy()
    total_value = current["value"].sum()
    current["weight"] = current["value"] / total_value if total_value else 0.0

    if previous is None or previous.empty:
        current["prev_shares"] = None
        current["share_change"] = None
        current["adjusted_change"] = None
        current["status"] = "NO_PRIOR"
        return current

    prev = previous[["cusip", "shares", "value", "name_of_issuer", "title_of_class"]].rename(
        columns={"shares": "prev_shares", "value": "prev_value", "name_of_issuer": "prev_name", "title_of_class": "prev_class"}
    )
    merged = current.merge(prev, on="cusip", how="outer")

    both = merged["shares"].gt(0) & merged["prev_shares"].gt(0)
    share_ratio = merged["shares"] / merged["prev_shares"]
    price_ratio = (merged["value"] / merged["shares"]) / (merged["prev_value"] / merged["prev_shares"])
    split = pd.Series(1.0, index=merged.index)
    split[both] = [_split_factor(s, p) for s, p in zip(share_ratio[both], price_ratio[both])]
    adjusted_ratio = share_ratio / split

    median_ratio = float(adjusted_ratio[both].median()) if both.any() else 1.0
    merged["share_change"] = share_ratio - 1.0
    merged["adjusted_change"] = adjusted_ratio / median_ratio - 1.0
    merged["split_factor"] = split

    status = pd.Series("UNCHANGED", index=merged.index)
    status[merged["adjusted_change"] > CHANGE_THRESHOLD] = "ADDED"
    status[merged["adjusted_change"] < -CHANGE_THRESHOLD] = "REDUCED"
    status[merged["prev_shares"].isna() | merged["prev_shares"].eq(0)] = "NEW"
    exited = merged["shares"].isna() | merged["shares"].eq(0)
    status[exited] = "EXITED"
    merged["status"] = status

    merged.loc[exited, "name_of_issuer"] = merged.loc[exited, "prev_name"]
    merged.loc[exited, "title_of_class"] = merged.loc[exited, "prev_class"]
    merged.loc[exited, ["shares", "value", "weight"]] = 0.0
    merged.loc[merged["status"].isin(["NEW", "EXITED"]), "adjusted_change"] = None
    return merged.drop(columns=["prev_value", "prev_name", "prev_class"]).sort_values("value", ascending=False).reset_index(drop=True)


def select_positions(changes: pd.DataFrame, previous: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Top holdings by value, plus the biggest adds/new positions and cuts/exits.

    Movers are only picked among meaningful positions (the MOVER_CANDIDATE_POOL
    largest this quarter, or last quarter for exits) -- a +400% change in a
    $2M stake of a $3T manager is noise, not conviction.
    """
    held = changes[changes["status"] != "EXITED"]
    top = held.head(TOP_N_BY_VALUE)
    pool = held.head(MOVER_CANDIDATE_POOL)

    adds = pool[pool["status"].isin(["NEW", "ADDED"])].copy()
    adds["_rank"] = adds["adjusted_change"].fillna(float("inf"))  # NEW positions first, then largest adds
    adds = adds.sort_values(["_rank", "value"], ascending=False).head(TOP_N_MOVERS).drop(columns="_rank")

    cuts = pool[pool["status"] == "REDUCED"]
    if previous is not None and not previous.empty:
        prev_pool = set(previous.head(MOVER_CANDIDATE_POOL)["cusip"])
        exits = changes[(changes["status"] == "EXITED") & changes["cusip"].isin(prev_pool)]
        cuts = pd.concat([exits, cuts])
    cuts = cuts.copy()
    cuts["_rank"] = cuts["adjusted_change"].fillna(-float("inf"))
    cuts = cuts.sort_values("_rank").head(TOP_N_MOVERS).drop(columns="_rank")

    selected = pd.concat([top, adds, cuts]).drop_duplicates(subset="cusip")
    return selected.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Mentions
# ----------------------------------------------------------------------------

_STATUS_TO_DIRECTION = {
    "NEW": ("POSITIVE", "High"),
    "ADDED": ("POSITIVE", "Medium"),
    "REDUCED": ("NEGATIVE", "Medium"),
    "EXITED": ("NEGATIVE", "High"),
    "UNCHANGED": ("MENTIONED", "Low"),
    "NO_PRIOR": ("MENTIONED", "Low"),
}


def _fmt_money(value: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= div:
            return f"${value / div:,.1f}{unit}"
    return f"${value:,.0f}"


def _describe(row: pd.Series, rank: Optional[int]) -> str:
    status = row["status"]
    if status == "EXITED":
        return f"Exited the position (held {row['prev_shares']:,.0f} shares last quarter)."
    base = f"Holds {row['shares']:,.0f} shares ({_fmt_money(row['value'])}, {row['weight']:.2%} of reported 13F portfolio"
    base += f", #{rank} by value)." if rank else ")."
    if status == "NEW":
        return base + " New position this quarter."
    if status == "NO_PRIOR":
        return base + " No prior quarter available to compare."
    change = f" Shares {row['share_change']:+.1%} vs prior quarter ({row['adjusted_change']:+.1%} relative to the filer's median change"
    if row.get("split_factor", 1.0) != 1.0:
        change += f", split-adjusted x{row['split_factor']:g}"
    change += ")."
    return base + change


def build_mentions(
    institution: str,
    filing: Filing13F,
    selected: pd.DataFrame,
    resolutions: dict[str, Optional[dict]],
    all_changes: pd.DataFrame,
) -> list[dict]:
    rank_by_cusip = {c: i + 1 for i, c in enumerate(all_changes[all_changes["status"] != "EXITED"]["cusip"])}
    source_path = str(_cache_path(filing))
    report_title = f"13F-HR holdings as of {filing.report_date} (filed {filing.filing_date})"

    mentions = []
    for _, row in selected.iterrows():
        resolution = resolutions.get(row["cusip"]) or {}
        if resolution.get("security_type") in EXCLUDED_SECURITY_TYPES:
            continue  # ETFs/funds held by the filer aren't companies
        direction, confidence = _STATUS_TO_DIRECTION[row["status"]]
        mentions.append(
            {
                "institution": institution,
                "report_title": report_title,
                "publication_date": filing.filing_date,
                "report_url": filing.index_url,
                "report_type": "13F",
                "investment_horizon": None,
                "theme": None,
                "region": None,
                "sector": None,
                "industry": None,
                "asset_class": "US equity",
                "company_name": row["name_of_issuer"],
                "ticker": (resolution.get("ticker") or None),
                "isin": None,
                "institutional_view": _describe(row, rank_by_cusip.get(row["cusip"])),
                "view_direction": direction,
                "confidence": confidence,
                "evidence": f"13F information table, CUSIP {row['cusip']} ({row['title_of_class']}), status {row['status']}",
                "source_path": source_path,
                "chunk_id": f"13F:{filing.cik}:{filing.accession}:{row['cusip']}",
            }
        )
    return mentions


def refresh_filer(
    institution: str,
    cik: int,
    fetch: Fetcher = sec_get,
    resolve: Optional[Callable[[list[str]], dict[str, Optional[dict]]]] = None,
) -> tuple[list[dict], dict]:
    """Latest quarter's mentions for one filer, plus a status summary row."""
    if resolve is None:
        import security_master

        resolve = security_master.resolve_cusips

    filings = list_13f_filings(cik, fetch)
    if not filings:
        return [], {"institution": institution, "cik": cik, "status": "no 13F-HR found"}

    current_filing = filings[0]
    current = load_holdings(current_filing, fetch)
    previous = load_holdings(filings[1], fetch) if len(filings) > 1 else None

    changes = compare_quarters(current, previous)
    selected = select_positions(changes, previous)
    resolutions = resolve(selected["cusip"].tolist())
    mentions = build_mentions(institution, current_filing, selected, resolutions, changes)

    resolved = sum(1 for m in mentions if m["ticker"])
    return mentions, {
        "institution": institution,
        "cik": cik,
        "status": "ok",
        "quarter": current_filing.report_date,
        "filed": current_filing.filing_date,
        "prior_quarter": filings[1].report_date if len(filings) > 1 else None,
        "positions_in_filing": len(current),
        "positions_selected": len(mentions),
        "tickers_resolved": resolved,
    }


def refresh_13f_mentions(
    filers: Optional[dict[str, int]] = None,
    fetch: Fetcher = sec_get,
    resolve: Optional[Callable[[list[str]], dict[str, Optional[dict]]]] = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch every filer's latest 13F -> mention rows. One failing filer never stops the others."""
    from institutional_research.schemas import OUTPUT_COLUMNS

    filers = filers or load_filers()
    all_mentions: list[dict] = []
    summaries: list[dict] = []
    for institution, cik in filers.items():
        try:
            mentions, summary = refresh_filer(institution, cik, fetch, resolve)
        except Exception as exc:  # network, parsing, SEC outage -- report, don't crash the whole refresh
            logger.exception("13F refresh failed for %s (CIK %s)", institution, cik)
            mentions, summary = [], {"institution": institution, "cik": cik, "status": f"error: {exc}"}
        logger.info("13F %s: %s", institution, summary)
        all_mentions.extend(mentions)
        summaries.append(summary)

    frame = pd.DataFrame(all_mentions, columns=OUTPUT_COLUMNS)
    return frame, summaries


def merge_13f_mentions(new_13f: pd.DataFrame, output_path: Path = None) -> pd.DataFrame:
    """Replace previously stored 13F rows for the refreshed institutions with the new ones.

    Report-derived (LLM) mentions are left untouched. Only the latest quarter
    per filer is kept -- an old quarter's holdings aren't a current view.
    """
    output_path = output_path or config.INSTITUTIONAL_MENTIONS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        refreshed = set(new_13f["institution"].unique()) if not new_13f.empty else set()
        if "report_type" in existing.columns:
            is_stale_13f = existing["report_type"].eq("13F") & existing["institution"].isin(refreshed)
            existing = existing[~is_stale_13f]
        combined = pd.concat([existing, new_13f], ignore_index=True) if not new_13f.empty else existing
    else:
        combined = new_13f
    combined.to_parquet(output_path, index=False)
    return combined


# Bump when a code change means previously built 13F mentions should be rebuilt
# (e.g. v2: CUSIP -> ticker resolution fixed for BRK/B-style and foreign lines).
REFRESH_VERSION = "2"


def _marker() -> Path:
    return config.RAW_DATA_DIR / "13f" / "_last_refresh.txt"


def last_refreshed_at() -> Optional[str]:
    marker = _marker()
    return marker.read_text().strip().split("|")[0] if marker.exists() else None


def _refresh_version() -> Optional[str]:
    marker = _marker()
    if not marker.exists():
        return None
    parts = marker.read_text().strip().split("|")
    return parts[1] if len(parts) > 1 else "1"


def mark_refreshed() -> None:
    marker = _marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}|{REFRESH_VERSION}")


AUTO_REFRESH_MAX_AGE_DAYS = 7  # new 13Fs only appear quarterly; weekly is plenty


def needs_refresh(max_age_days: int = AUTO_REFRESH_MAX_AGE_DAYS) -> bool:
    """True if 13F data was never fetched, or the last refresh is older than `max_age_days`."""
    last = last_refreshed_at()
    if not last or _refresh_version() != REFRESH_VERSION:
        return True
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(last)
    except ValueError:
        return True
    return age.days >= max_age_days
