"""Insider open-market buying and selling (SEC Form 4) as point-in-time features.

Why this exists: 13F holdings say what large managers own, with a 45-day
lag. A Form 4 says what a company's own officers and directors did with
their own money, filed within 2 business days. Open-market *purchases* are
the classic informative case -- insiders sell for many reasons (taxes,
diversification, planned 10b5-1 sales), but buy for essentially one.

Only two transaction codes are kept, and only for officers/directors:
    P  open-market or private purchase
    S  open-market or private sale
Everything else (option exercises M, tax withholding F, grants A, gifts G,
...) is compensation plumbing, not a decision to buy or sell. 10%-owners who
are not officers/directors are dropped: they are usually funds.

Two sources produce the same normalized transaction table (TRANSACTION_COLUMNS):

- **per-issuer** (`fetch_issuer_transactions`): ticker -> issuer CIK via SEC
  company_tickers.json, then the issuer's submissions JSON, then each Form 4's
  XML (`parse_form4_xml`), cached per accession. Exact, but one request per
  filing -- fine for a handful of tickers, ~300k requests (~13 hours at the SEC
  fair-access rate) for a 1,000-name, 7-year backtest.
- **bulk** (`load_bulk_transactions`): the SEC's quarterly "Insider
  Transactions Data Sets" -- the same Form 3/4/5 fields, one zip per quarter
  (~13 MB), so ~30 requests for the whole backtest. Parsed and filtered once,
  cached as parquet per quarter.

Point-in-time rule: a transaction is usable from its SEC FILING date, never
its transaction date (up to 2 business days earlier, much more for late filers).
"""

from __future__ import annotations

import io
import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

import config
from institutional_research.holdings_13f import SUBMISSIONS_PAGE_URL, SUBMISSIONS_URL, Fetcher, sec_get

logger = logging.getLogger(__name__)

FORM4_DIR = config.RAW_DATA_DIR / "form4"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
BULK_INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
SEC_BASE_URL = "https://www.sec.gov"
FORM4_XML_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{name}"

KEPT_CODES = {"P", "S"}
WINDOW_DAYS = 90
CLUSTER_MIN_BUYERS = 3

TRANSACTION_COLUMNS = [
    "accession", "filing_date", "issuer_cik", "owner_cik", "is_director", "is_officer",
    "trans_code", "trans_date", "shares", "price", "value",
]
FEATURE_COLUMNS = ["insider_buyers_90d", "insider_net_buy_value_90d_mcap", "insider_cluster_buy"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)


def _cik(value) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Ticker -> issuer CIK
# ----------------------------------------------------------------------------


def ticker_to_cik(fetch: Fetcher = sec_get, refresh: bool = False) -> dict[str, int]:
    """SEC's current ticker -> CIK map (cached on disk).

    Today's map: a delisted ticker is missing (its prices are usually missing
    too), and a ticker reused by another company maps to the new one.
    """
    path = FORM4_DIR / "company_tickers.json"
    if path.exists() and not refresh:
        raw = json.loads(path.read_text())
    else:
        raw = json.loads(fetch(COMPANY_TICKERS_URL))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
    # Yahoo writes share classes with "-" (BRK-B); SEC with "-" too, but some use ".".
    return {str(v["ticker"]).upper().replace(".", "-"): int(v["cik_str"]) for v in raw.values()}


# ----------------------------------------------------------------------------
# Per-filing XML (per-issuer source)
# ----------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(element: Optional[ET.Element], *path: str) -> Optional[ET.Element]:
    """Namespace-agnostic nested child lookup."""
    for name in path:
        if element is None:
            return None
        element = next((c for c in element if _local(c.tag) == name), None)
    return element


def _text(element: Optional[ET.Element], *path: str) -> Optional[str]:
    """Text at `path`, reading through Form 4's <value> wrappers."""
    node = _find(element, *path)
    if node is None:
        return None
    value = _find(node, "value")
    text = (value if value is not None else node).text
    return text.strip() if text and text.strip() else None


def _flag(text: Optional[str]) -> bool:
    return (text or "").strip().lower() in {"1", "true"}


def _float(text: Optional[str]) -> Optional[float]:
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_form4_xml(xml_bytes: bytes, accession: str, filing_date: str) -> pd.DataFrame:
    """Officer/director open-market P/S non-derivative transactions of one Form 4.

    `filing_date` comes from the submissions index (it is not in the XML).
    A filing with several reporting owners (joint filing) lists each owner:
    each transaction row is repeated per qualifying owner, and `value` is
    counted once per transaction when aggregating (see `insider_features`).
    """
    root = ET.fromstring(xml_bytes)
    issuer_cik = _cik(_text(root, "issuer", "issuerCik"))

    owners = []
    for owner in (c for c in root if _local(c.tag) == "reportingOwner"):
        relationship = _find(owner, "reportingOwnerRelationship")
        owners.append({
            "owner_cik": _cik(_text(owner, "reportingOwnerId", "rptOwnerCik")),
            "is_director": _flag(_text(relationship, "isDirector")),
            "is_officer": _flag(_text(relationship, "isOfficer")),
        })
    owners = [o for o in owners if o["is_director"] or o["is_officer"]]

    table = _find(root, "nonDerivativeTable")
    rows = []
    for trans in ([] if table is None else [c for c in table if _local(c.tag) == "nonDerivativeTransaction"]):
        code = (_text(trans, "transactionCoding", "transactionCode") or "").upper()
        if code not in KEPT_CODES:
            continue
        shares = _float(_text(trans, "transactionAmounts", "transactionShares"))
        price = _float(_text(trans, "transactionAmounts", "transactionPricePerShare"))
        for owner in owners:
            rows.append({
                "accession": accession, "filing_date": filing_date, "issuer_cik": issuer_cik, **owner,
                "trans_code": code, "trans_date": _text(trans, "transactionDate"),
                "shares": shares, "price": price,
                "value": shares * price if shares is not None and price is not None else None,
            })
    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS) if rows else _empty()


def _raw_xml_name(primary_document: str) -> str:
    # submissions JSON points at the XSL-rendered view ("xslF345X05/form4.xml"); the raw XML sits one level up.
    return primary_document.split("/")[-1]


def _form4_filings(block: dict) -> list[tuple[str, str, str]]:
    forms = block.get("form", [])
    return [
        (block["accessionNumber"][i], block["filingDate"][i], block["primaryDocument"][i])
        for i, form in enumerate(forms) if form == "4"
    ]


def fetch_issuer_transactions(cik: int, since: str, fetch: Fetcher = sec_get) -> pd.DataFrame:
    """Every Form 4 about one issuer filed since `since`, parsed and cached per accession."""
    data = json.loads(fetch(SUBMISSIONS_URL.format(cik=cik)))
    filings = _form4_filings(data.get("filings", {}).get("recent", {}))
    for page in data.get("filings", {}).get("files", []):
        if page.get("filingTo") and page["filingTo"] < since:
            continue
        filings += _form4_filings(json.loads(fetch(SUBMISSIONS_PAGE_URL.format(name=page["name"]))))

    frames = []
    for accession, filing_date, primary in filings:
        if filing_date < since:
            continue
        path = FORM4_DIR / "filings" / str(cik) / f"{accession.replace('-', '')}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        url = FORM4_XML_URL.format(cik=cik, acc=accession.replace("-", ""), name=_raw_xml_name(primary))
        try:
            parsed = parse_form4_xml(fetch(url), accession, filing_date)
        except Exception as exc:  # one malformed filing never drops the issuer
            logger.warning("Form 4 %s (CIK %s) skipped: %s", accession, cik, exc)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        parsed.to_parquet(path, index=False)  # cached even when empty: a filed Form 4 never changes
        frames.append(parsed)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else _empty()


# ----------------------------------------------------------------------------
# Quarterly bulk data sets (bulk source)
# ----------------------------------------------------------------------------


def bulk_dataset_urls(fetch: Fetcher = sec_get) -> dict[str, str]:
    """{"2025q1": absolute zip URL} scraped from the SEC data-set page (paths vary by year)."""
    html = fetch(BULK_INDEX_URL).decode("utf-8", errors="replace")
    urls = {}
    for href in re.findall(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', html):
        urls[href[1]] = href[0] if href[0].startswith("http") else SEC_BASE_URL + href[0]
    return urls


def parse_bulk_zip(content: bytes) -> pd.DataFrame:
    """One quarterly zip -> officer/director P/S transactions from original Form 4s."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        def read(name, cols):
            with archive.open(name) as handle:
                return pd.read_csv(handle, sep="\t", usecols=cols, dtype=str, quoting=3, on_bad_lines="skip")

        subs = read("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])
        owners = read("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        trans = read("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])

    subs = subs[subs["DOCUMENT_TYPE"] == "4"]  # amendments ignored, as for 13F
    trans = trans[trans["TRANS_CODE"].str.upper().isin(KEPT_CODES)]
    relationship = owners["RPTOWNER_RELATIONSHIP"].fillna("")
    owners = owners.assign(is_director=relationship.str.contains("Director"), is_officer=relationship.str.contains("Officer"))
    owners = owners[owners["is_director"] | owners["is_officer"]]

    merged = trans.merge(subs, on="ACCESSION_NUMBER").merge(owners, on="ACCESSION_NUMBER")
    if merged.empty:
        return _empty()
    shares = pd.to_numeric(merged["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(merged["TRANS_PRICEPERSHARE"], errors="coerce")
    return pd.DataFrame({
        "accession": merged["ACCESSION_NUMBER"],
        "filing_date": pd.to_datetime(merged["FILING_DATE"], format="%d-%b-%Y", errors="coerce").dt.strftime("%Y-%m-%d"),
        "issuer_cik": pd.to_numeric(merged["ISSUERCIK"], errors="coerce").astype("Int64"),
        "owner_cik": pd.to_numeric(merged["RPTOWNERCIK"], errors="coerce").astype("Int64"),
        "is_director": merged["is_director"],
        "is_officer": merged["is_officer"],
        "trans_code": merged["TRANS_CODE"].str.upper(),
        "trans_date": merged["TRANS_DATE"],
        "shares": shares,
        "price": price,
        "value": shares * price,
    }, columns=TRANSACTION_COLUMNS)


def _quarters(start: str, end: str) -> list[str]:
    periods = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="Q")
    return [f"{p.year}q{p.quarter}" for p in periods]


def load_bulk_transactions(start: str, end: str, fetch: Fetcher = sec_get) -> tuple[pd.DataFrame, Optional[str]]:
    """All officer/director P/S Form 4 transactions filed in [start, end], from the quarterly data sets.

    Returns (transactions, coverage_end): the last day of the last quarter
    actually available. Dates after it get no insider features (NaN), not
    zeros -- "no data yet" is not "no insider bought".
    """
    wanted = _quarters(start, end)
    cache_dir = FORM4_DIR / "bulk"
    urls: Optional[dict[str, str]] = None
    frames, covered = [], []
    for quarter in wanted:
        path = cache_dir / f"{quarter}.parquet"
        if not path.exists():
            if urls is None:
                urls = bulk_dataset_urls(fetch)
            if quarter not in urls:
                logger.info("Form 4 data set %s not published yet", quarter)
                continue
            parsed = parse_bulk_zip(fetch(urls[quarter]))
            path.parent.mkdir(parents=True, exist_ok=True)
            parsed.to_parquet(path, index=False)
        frames.append(pd.read_parquet(path))
        covered.append(quarter)
    if not covered:
        return _empty(), None
    last = pd.Period(covered[-1].replace("q", "Q"), freq="Q").end_time.normalize()
    frames = [f for f in frames if not f.empty]
    return (pd.concat(frames, ignore_index=True) if frames else _empty()), str(last.date())


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------


def insider_features(
    keys: pd.DataFrame,
    transactions: pd.DataFrame,
    ticker_ciks: dict[str, int],
    coverage_end: Optional[str] = None,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Per (date, ticker): insider activity filed in (date - window, date].

    `keys` needs columns date, ticker, and optionally log_market_cap.
      insider_buyers_90d             distinct officer/director buyers (code P)
      insider_net_buy_value_90d_mcap (P value - S value) / market cap
      insider_cluster_buy            1 if >= 3 distinct buyers, else 0
    No transactions = 0 (the data covers every filing). NaN when the ticker
    has no issuer CIK, the date is past `coverage_end`, or (for the value
    ratio) market cap is unknown.
    """
    out = keys[["date", "ticker"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in FEATURE_COLUMNS:
        out[col] = np.nan
    if out.empty:
        return out

    tx = transactions.copy()
    tx["filing_date"] = pd.to_datetime(tx["filing_date"])
    tx["issuer_cik"] = pd.to_numeric(tx["issuer_cik"], errors="coerce")
    # value counted once per transaction, not once per joint-filing owner
    tx_values = tx.drop_duplicates(subset=["accession", "trans_code", "trans_date", "shares", "price"])
    by_issuer = {cik: g for cik, g in tx.groupby("issuer_cik")}
    values_by_issuer = {cik: g for cik, g in tx_values.groupby("issuer_cik")}
    empty = tx.iloc[0:0]
    window = pd.Timedelta(days=window_days)
    end = pd.Timestamp(coverage_end) if coverage_end else None
    market_cap = np.exp(keys["log_market_cap"].to_numpy(dtype=float)) if "log_market_cap" in keys.columns else None

    buyers, net_values, clusters = [], [], []
    for i, (date, ticker) in enumerate(zip(out["date"], out["ticker"])):
        cik = ticker_ciks.get(str(ticker).upper())
        if cik is None or (end is not None and date > end):
            buyers.append(np.nan)
            net_values.append(np.nan)
            clusters.append(np.nan)
            continue
        g = by_issuer.get(cik, empty)
        in_window = g[(g["filing_date"] <= date) & (g["filing_date"] > date - window)]
        n_buyers = in_window.loc[in_window["trans_code"] == "P", "owner_cik"].nunique()
        v = values_by_issuer.get(cik, empty)
        v = v[(v["filing_date"] <= date) & (v["filing_date"] > date - window)]
        net = v.loc[v["trans_code"] == "P", "value"].sum() - v.loc[v["trans_code"] == "S", "value"].sum()
        mcap = market_cap[i] if market_cap is not None else np.nan
        buyers.append(float(n_buyers))
        net_values.append(float(net) / mcap if np.isfinite(mcap) and mcap > 0 else np.nan)
        clusters.append(float(n_buyers >= CLUSTER_MIN_BUYERS))
    out["insider_buyers_90d"] = buyers
    out["insider_net_buy_value_90d_mcap"] = net_values
    out["insider_cluster_buy"] = clusters
    return out


def load_transactions(
    tickers: Iterable[str], start: str, end: str, source: str = "bulk", fetch: Fetcher = sec_get
) -> tuple[pd.DataFrame, dict[str, int], Optional[str]]:
    """(transactions, ticker -> CIK, coverage_end) from the chosen source."""
    ciks = ticker_to_cik(fetch)
    if source == "bulk":
        transactions, coverage_end = load_bulk_transactions(start, end, fetch)
        return transactions, ciks, coverage_end
    if source != "per-issuer":
        raise ValueError(f"source must be 'bulk' or 'per-issuer', got {source!r}")
    frames = []
    for ticker in sorted({t.upper() for t in tickers}):
        cik = ciks.get(ticker)
        if cik is None:
            continue
        try:
            frames.append(fetch_issuer_transactions(cik, start, fetch))
        except Exception as exc:
            logger.warning("Form 4 fetch failed for %s (CIK %s): %s", ticker, cik, exc)
            ciks = {k: v for k, v in ciks.items() if k != ticker}  # unknown, not "no insider activity"
    frames = [f for f in frames if not f.empty]
    return (pd.concat(frames, ignore_index=True) if frames else _empty()), ciks, None
