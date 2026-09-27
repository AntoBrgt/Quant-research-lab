"""Point-in-time fundamentals from SEC XBRL companyfacts (PHASE 1).

Why not yfinance: its statements cover ~4-5 fiscal years and are *as
currently restated*; nothing before ~2021 can be tested, and a restated
number is not what the market saw. Every XBRL fact carries the date the
filing that reported it was **filed**, so a value is usable from that date
and not a day earlier -- the definition of point-in-time. Coverage: 10-K/10-Q
XBRL from 2009 (large filers) / 2011 (everyone).

Source: the SEC's nightly bulk `companyfacts.zip` (one JSON per company,
~1.4 GB), read one CIK at a time and cached as a small parquet per CIK.

Rules:
- **First-filed value wins.** A period restated in a later filing keeps the
  value first reported; the restatement only exists from its own filing date
  (not modelled: we keep the original, which is what traders saw first).
- **Discrete quarters.** 10-Qs report 3-month values (and year-to-date);
  10-Ks report the fiscal year. Q4 = fiscal year - (Q1 + Q2 + Q3), filed on
  the 10-K's date. EPS Q4 derived the same way is an approximation
  (share counts differ across quarters) and is flagged as derived.
- **Instants** (shares outstanding, debt, cash) are taken as of the latest
  balance-sheet date filed by the as-of date.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

XBRL_DIR = config.RAW_DATA_DIR / "sec_xbrl"
COMPANYFACTS_ZIP = XBRL_DIR / "companyfacts.zip"
PIT_DIR = XBRL_DIR / "pit"
CACHE_VERSION = "v4"

# Tag fallbacks, in order of preference (companies switch tags over time, e.g. ASC 606 in 2018).
DURATION_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"],
}
INSTANT_TAGS = {
    "shares": [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")],
    "cash": [("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
             ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"), ("us-gaap", "Cash")],
    "debt_total": [("us-gaap", "LongTermDebt"), ("us-gaap", "DebtLongtermAndShorttermCombinedAmount")],
    "debt_noncurrent": [("us-gaap", "LongTermDebtNoncurrent")],
    "debt_current": [("us-gaap", "LongTermDebtCurrent"), ("us-gaap", "DebtCurrent")],
    "short_borrowings": [("us-gaap", "ShortTermBorrowings")],
    "assets": [("us-gaap", "Assets")],
    "equity": [("us-gaap", "StockholdersEquity"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
}
QUARTER_DAYS = (80, 100)
YEAR_DAYS = (350, 380)
NINE_MONTH_DAYS = (260, 285)


# ----------------------------------------------------------------------------
# Raw facts
# ----------------------------------------------------------------------------


_ARCHIVE: dict[str, zipfile.ZipFile] = {}


def _archive(zip_path: Path) -> zipfile.ZipFile:
    """Open the 1.4 GB bulk zip once per process (reading its directory is the slow part)."""
    key = str(zip_path)
    if key not in _ARCHIVE:
        _ARCHIVE[key] = zipfile.ZipFile(zip_path)
    return _ARCHIVE[key]


def read_companyfacts(cik: int, zip_path: Path = None) -> Optional[dict]:
    zip_path = zip_path or COMPANYFACTS_ZIP
    try:
        with _archive(zip_path).open(f"CIK{int(cik):010d}.json") as handle:
            return json.load(handle)
    except KeyError:
        return None


def _facts(doc: dict, namespace: str, tag: str) -> list[dict]:
    units = doc.get("facts", {}).get(namespace, {}).get(tag, {}).get("units", {})
    out = []
    for unit, rows in units.items():
        for r in rows:
            if r.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F", "10-KT", "10-QT"):
                continue
            out.append({**r, "unit": unit})
    return out


def _first_filed(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """For each period, the value as FIRST filed (restatements come later and are ignored)."""
    return frame.sort_values("filed").drop_duplicates(subset=keys, keep="first")


def duration_facts(doc: dict, tags: list[str]) -> pd.DataFrame:
    """Duration facts from ALL fallback tags (companies switch tags, e.g. ASC 606 in 2018, or
    Apple's OCF moving from ...ContinuingOperations in 2017). Per period, the EARLIEST-FILED value
    wins, whatever its tag -- the later tag often only appears as a restated comparative a year
    later; tag preference only breaks same-day ties."""
    frames = []
    for rank, tag in enumerate(tags):
        rows = [r for r in _facts(doc, "us-gaap", tag) if r.get("start")]
        if rows:
            frames.append(pd.DataFrame(rows).assign(tag_rank=rank))
    if not frames:
        return pd.DataFrame(columns=["start", "end", "days", "val", "filed", "form"])
    frame = pd.concat(frames, ignore_index=True)
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame["days"] = (frame["end"] - frame["start"]).dt.days
    frame = frame.sort_values(["filed", "tag_rank"]).drop_duplicates(subset=["start", "end"], keep="first")
    return frame[["start", "end", "days", "val", "filed", "form"]].reset_index(drop=True)


def discrete_quarters(facts: pd.DataFrame) -> pd.DataFrame:
    """Quarter values (end, val, filed, derived).

    - An explicit 3-month fact is used as is.
    - Otherwise a quarter is the difference of two consecutive cumulative
      (year-to-date) facts with the same start: 6M - 3M, 9M - 6M, FY - 9M.
      Cash-flow statements are only reported year-to-date, so this is how
      quarterly operating cash flow and capex exist at all.
    The derived quarter is available from the later of the two filing dates.
    """
    if facts.empty:
        return pd.DataFrame(columns=["end", "val", "filed", "derived"])
    quarters = {r.end: (float(r.val), r.filed, False) for r in facts[facts["days"].between(*QUARTER_DAYS)].itertuples()}
    cumulative = facts[facts["days"] >= QUARTER_DAYS[0]]
    for _, group in cumulative.groupby("start"):
        group = group.sort_values("end")
        rows = list(group.itertuples())
        for prev, cur in zip(rows, rows[1:]):
            gap = (cur.end - prev.end).days
            if cur.end in quarters or not (QUARTER_DAYS[0] <= gap <= QUARTER_DAYS[1]):
                continue
            quarters[cur.end] = (float(cur.val) - float(prev.val), max(cur.filed, prev.filed), True)
    out = pd.DataFrame([(e, v, f, d) for e, (v, f, d) in quarters.items()], columns=["end", "val", "filed", "derived"])
    return out.sort_values("end").reset_index(drop=True)


def instant_facts(doc: dict, tags: list[tuple[str, str]]) -> pd.DataFrame:
    """Instant facts from all fallback tags; per date the earliest-filed value wins (tag order breaks ties)."""
    frames = []
    for rank, (namespace, tag) in enumerate(tags):
        rows = [r for r in _facts(doc, namespace, tag) if not r.get("start")]
        if rows:
            frames.append(pd.DataFrame(rows).assign(tag_rank=rank))
    if not frames:
        return pd.DataFrame(columns=["end", "val", "filed"])
    frame = pd.concat(frames, ignore_index=True)
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame["filed"])
    frame = frame.sort_values(["filed", "tag_rank"]).drop_duplicates(subset=["end"], keep="first")
    return frame[["end", "val", "filed"]].sort_values("end").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Per-company PIT table (cached)
# ----------------------------------------------------------------------------


def company_table(cik: int, zip_path: Path = None, refresh: bool = False) -> pd.DataFrame:
    """Long table: kind ('q' quarterly / 'i' instant), field, end, val, filed, derived.

    Cached per CIK; the bulk zip is only opened when the cache is missing.
    """
    path = PIT_DIR / CACHE_VERSION / f"{int(cik)}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    doc = read_companyfacts(cik, zip_path)
    frames = []
    if doc:
        for field, tags in DURATION_TAGS.items():
            q = discrete_quarters(duration_facts(doc, tags))
            if len(q):
                frames.append(q.assign(kind="q", field=field))
        for field, tags in INSTANT_TAGS.items():
            i = instant_facts(doc, tags)
            if len(i):
                frames.append(i.assign(kind="i", field=field, derived=False))
    table = (pd.concat(frames, ignore_index=True) if frames else
             pd.DataFrame(columns=["end", "val", "filed", "derived", "kind", "field"]))
    table["val"] = pd.to_numeric(table["val"], errors="coerce").astype(float)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


# ----------------------------------------------------------------------------
# As-of snapshots
# ----------------------------------------------------------------------------


def _visible(table: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    return table[table["filed"] <= as_of]


def quarterly_series(table: pd.DataFrame, field: str, as_of) -> pd.Series:
    """Discrete quarterly values of `field` known at `as_of`, indexed by period end."""
    t = _visible(table, pd.Timestamp(as_of))
    t = t[(t["kind"] == "q") & (t["field"] == field)]
    return t.set_index("end")["val"].sort_index()


def _ttm(series: pd.Series) -> Optional[float]:
    """Sum of the last 4 quarters, only if they are 4 consecutive quarters (~1 year span)."""
    if len(series) < 4:
        return None
    last = series.iloc[-4:]
    span = (last.index[-1] - last.index[0]).days
    return float(last.sum()) if 250 <= span <= 300 else None


def _latest_instant(table: pd.DataFrame, field: str, as_of, max_age_days: int = 400) -> Optional[float]:
    t = _visible(table, pd.Timestamp(as_of))
    t = t[(t["kind"] == "i") & (t["field"] == field)]
    if t.empty:
        return None
    row = t.sort_values("end").iloc[-1]
    return float(row["val"]) if (pd.Timestamp(as_of) - row["end"]).days <= max_age_days else None


def snapshot(table: pd.DataFrame, as_of) -> dict:
    """TTM revenue / net income / FCF, net debt, shares -- as known at `as_of` (filing dates <= as_of)."""
    as_of = pd.Timestamp(as_of)
    rev = quarterly_series(table, "revenue", as_of)
    ni = quarterly_series(table, "net_income", as_of)
    ocf = quarterly_series(table, "ocf", as_of)
    capex = quarterly_series(table, "capex", as_of)
    fcf_q = (ocf - capex.reindex(ocf.index)).dropna() if len(ocf) and len(capex) else pd.Series(dtype=float)
    out = {
        "revenue_ttm": _ttm(rev),
        "net_income_ttm": _ttm(ni),
        "fcf_ttm": _ttm(fcf_q),
        "revenue_ttm_prior": _ttm(rev.iloc[:-4]) if len(rev) >= 8 else None,
        "shares": _latest_instant(table, "shares", as_of),
        "cash": _latest_instant(table, "cash", as_of),
        "equity": _latest_instant(table, "equity", as_of),
        "last_filed": _visible(table, as_of)["filed"].max() if len(_visible(table, as_of)) else None,
    }
    total = _latest_instant(table, "debt_total", as_of)
    if total is None:
        parts = [_latest_instant(table, f, as_of) for f in ("debt_noncurrent", "debt_current")]
        total = sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None
    short = _latest_instant(table, "short_borrowings", as_of)
    if total is None and _latest_instant(table, "assets", as_of) is not None:
        total = 0.0  # a filer with a balance sheet and no debt tag reports no debt
    out["debt"] = None if total is None else total + (short or 0.0)
    out["net_debt"] = None if out["debt"] is None or out["cash"] is None else out["debt"] - out["cash"]
    return out


def thesis_break_dates(table: pd.DataFrame) -> list[pd.Timestamp]:
    """Filing dates on which a thesis break becomes known.

    Break = two consecutive reported quarters, each with negative quarterly
    FCF AND revenue below the same quarter a year earlier. Known on the
    filing date of the second quarter (never its period end).
    """
    rev = table[(table["kind"] == "q") & (table["field"] == "revenue")].set_index("end").sort_index()
    ocf = table[(table["kind"] == "q") & (table["field"] == "ocf")].set_index("end").sort_index()
    capex = table[(table["kind"] == "q") & (table["field"] == "capex")].set_index("end").sort_index()
    if rev.empty or ocf.empty or capex.empty:
        return []
    frame = pd.DataFrame({"rev": rev["val"], "ocf": ocf["val"], "capex": capex["val"]}).dropna()
    filed = pd.concat([rev["filed"], ocf["filed"], capex["filed"]], axis=1).max(axis=1).reindex(frame.index)
    frame["fcf"] = frame["ocf"] - frame["capex"]
    ends = list(frame.index)
    bad = []
    for i, end in enumerate(ends):
        prior = [e for e in ends[:i] if 350 <= (end - e).days <= 380]
        bad.append(bool(prior) and frame.loc[end, "fcf"] < 0 and frame.loc[end, "rev"] < frame.loc[prior[-1], "rev"])
    return [filed[ends[i]] for i in range(1, len(ends)) if bad[i] and bad[i - 1]
            and 80 <= (ends[i] - ends[i - 1]).days <= 100]


def sue_series(table: pd.DataFrame, min_history: int = 8) -> pd.DataFrame:
    """Standardized unexpected earnings without analyst estimates, per quarter, with its filing date.

    SUE_q = (EPS_q - EPS_{q-4}) / std(last `min_history` such changes, ending at q-1).
    Available from the filing date of quarter q (never the period end).
    """
    eps = table[(table["kind"] == "q") & (table["field"] == "eps")].set_index("end").sort_index()
    if len(eps) < min_history + 5:
        return pd.DataFrame(columns=["end", "filed", "sue"])
    ends = list(eps.index)
    changes, rows = {}, []
    for i, end in enumerate(ends):
        prior = [e for e in ends[:i] if 350 <= (end - e).days <= 380]
        if prior:
            changes[end] = eps.loc[end, "val"] - eps.loc[prior[-1], "val"]
    keys = list(changes)
    for j, end in enumerate(keys):
        history = [changes[k] for k in keys[max(0, j - min_history):j]]
        if len(history) < min_history:
            continue
        sd = float(np.std(history, ddof=1))
        if sd > 0:
            rows.append({"end": end, "filed": eps.loc[end, "filed"], "sue": changes[end] / sd})
    return pd.DataFrame(rows, columns=["end", "filed", "sue"])


def sic_codes(ciks, fetch=None) -> dict[int, Optional[int]]:
    """SEC SIC industry code per CIK (from the submissions JSON), cached.

    Used as a coarse sector: today's code, not point-in-time -- a company's
    industry classification rarely changes, and it is only a feature.
    """
    if fetch is None:
        from institutional_research.holdings_13f import sec_get as fetch
    from institutional_research.holdings_13f import SUBMISSIONS_URL

    path = XBRL_DIR / "sic.json"
    cache = {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}
    missing = [int(c) for c in ciks if int(c) not in cache]
    for i, cik in enumerate(missing):
        try:
            sic = json.loads(fetch(SUBMISSIONS_URL.format(cik=cik))).get("sic")
            cache[cik] = int(sic) if sic not in (None, "") else None
        except Exception as exc:
            logger.warning("SIC lookup failed for CIK %s: %s", cik, exc)
            cache[cik] = None
        if i % 200 == 199:
            path.write_text(json.dumps(cache))
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return {int(c): cache.get(int(c)) for c in ciks}


def sic_division(sic: Optional[int]) -> str:
    """SIC major division (Manufacturing, Finance, ...) -- 10 coarse sectors."""
    if sic is None:
        return "unknown"
    for upper, name in ((999, "agriculture"), (1499, "mining"), (1799, "construction"), (3999, "manufacturing"),
                        (4999, "transport_utilities"), (5199, "wholesale"), (5999, "retail"),
                        (6799, "finance"), (8999, "services")):
        if sic <= upper:
            return name
    return "public_admin"
