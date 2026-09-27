"""Build the (date, ticker) research panel.

One row per rebalance date x ticker with:
- point-in-time technical + fundamental features (`pit_features.py`)
- baseline scores, computed by the production `horizon.compute_horizon_weighted_view`:
    score_full       technicals + fundamentals (the app's horizon score)
    score_technical  technicals only -- available over the whole price history,
                     whereas yfinance fundamentals only cover the last ~4 fiscal years
    score_rank       (universe="pit" only) score_full + INSTITUTIONAL_TILT x
                     point-in-time institutional_direction_score -- the app's
                     full `rank_score`, the thing its labels are cut from
- reference signals: momentum_12_1, low_volatility (= -volatility_60d)
- labels: fwd_return, fwd_excess (vs benchmark), from `labels.py`

Universe (STEP 11b):
- "current": a fixed ticker list, typically today's 13F universe applied to
  every past date. Survivorship-biased: today's holdings are, by
  construction, the names that survived and grew. The first real run showed
  it -- every score bucket "beat" SPY by ~5%/quarter.
- "pit": at each rebalance date, only tickers in the latest 13F filing of at
  least one filer with `filing_date <= date` (SEC filing date, never the
  quarter end). Labels are entered at the NEXT day's close (`labels.py`), so
  a filing made after the close of its filing date is still tradable.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

import horizon
import screener
from backtest import labels, pit_features
from institutional_research import holdings_13f
from institutional_research import universe as inst_universe

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_DAYS = 91  # "3 months" preset -> MEDIUM profile
DEFAULT_LABEL_DAYS = 63  # ~3 months of trading days
DEFAULT_REBALANCE_EVERY = 21  # ~monthly
WARMUP_DAYS = 252  # the 252-day return / 200-day MA need a year of history
# A filer's latest filing stops counting once it is this old: 13Fs are due
# 45 days after each quarter end, so consecutive filings are ~91 days apart
# (at most ~135). A filer that stopped filing (merged, new CIK, fell below
# $100M) would otherwise keep its last holdings in the universe forever.
PIT_MAX_FILING_AGE_DAYS = 200

UNIVERSES = ("current", "pit", "pit_broad")
MEMBERSHIP_COLUMNS = ["date", "ticker", "cusip", "institution_count", "institutional_direction_score"]


def rebalance_dates(calendar: pd.DatetimeIndex, every: int, warmup: int = WARMUP_DAYS, start=None, end=None) -> list[pd.Timestamp]:
    dates = list(calendar.sort_values()[warmup::every])
    if start is not None:
        dates = [d for d in dates if d >= pd.Timestamp(start)]
    if end is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    return dates


def pit_universe(
    history: pd.DataFrame, dates: Iterable, max_filing_age_days: int = PIT_MAX_FILING_AGE_DAYS
) -> pd.DataFrame:
    """Point-in-time universe membership: one row per (date, member).

    `history` is `holdings_13f.build_universe_history` output. For each date,
    each filer contributes the rows of its latest filing with
    `filing_date <= date` (and not older than `max_filing_age_days`). A
    ticker's `institutional_direction_score` is then aggregated exactly like
    the live universe (`universe.aggregate_direction` over its rows).

    Members whose CUSIP never resolved to a ticker are kept, one row per
    (date, cusip) with ticker=None, so coverage can count them.
    """
    if history is None or history.empty:
        return pd.DataFrame(columns=MEMBERSHIP_COLUMNS)
    history = history.assign(filing_date=pd.to_datetime(history["filing_date"]))
    filings = history[["cik", "accession", "filing_date"]].drop_duplicates().sort_values("filing_date")
    has_ticker = history["ticker"].notna() & history["ticker"].astype(str).str.strip().ne("")

    rows = []
    for date in sorted(pd.Timestamp(d) for d in dates):
        window = filings[(filings["filing_date"] <= date)
                         & (filings["filing_date"] >= date - pd.Timedelta(days=max_filing_age_days))]
        if window.empty:
            continue
        latest = history["accession"].isin(set(window.groupby("cik").tail(1)["accession"]))

        for ticker, group in history[latest & has_ticker].groupby("ticker"):
            score, _ = inst_universe.aggregate_direction(group["view_direction"])
            rows.append({"date": date, "ticker": str(ticker).upper(), "cusip": group["cusip"].iloc[0],
                         "institution_count": group["institution"].nunique(),
                         "institutional_direction_score": score})
        for cusip, group in history[latest & ~has_ticker].groupby("cusip"):
            rows.append({"date": date, "ticker": None, "cusip": cusip,
                         "institution_count": group["institution"].nunique(),
                         "institutional_direction_score": None})
    return pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)


def universe_coverage(membership: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per year: how many PIT members the backtest could actually use.

    A member counts as priced on a date if the panel has a row for it (i.e.
    `pit_technicals` found a price within `MAX_STALE_DAYS`). Delisted,
    renamed and unresolved names show up here as the gap -- they are part of
    the true universe, and the backtest can't see how they performed.
    """
    if membership.empty:
        return pd.DataFrame()
    m = membership.copy()
    m["date"] = pd.to_datetime(m["date"])
    m["has_ticker"] = m["ticker"].notna()
    keys = list(zip(m["date"], m["ticker"]))
    priced = set(zip(panel["date"], panel["ticker"])) if not panel.empty else set()
    labelled_panel = panel.dropna(subset=["fwd_excess"]) if not panel.empty else panel
    labelled = set(zip(labelled_panel["date"], labelled_panel["ticker"])) if not panel.empty else set()
    m["has_price"] = [k in priced for k in keys]
    m["has_label"] = [k in labelled for k in keys]
    m["year"] = m["date"].dt.year

    out = m.groupby("year").agg(
        dates=("date", "nunique"),
        members=("date", "size"),
        pct_with_ticker=("has_ticker", "mean"),
        pct_priced=("has_price", "mean"),
        pct_labelled=("has_label", "mean"),
        tickers=("ticker", "nunique"),
    )
    out["avg_members_per_date"] = out.pop("members") / out["dates"]
    ever_priced = m[m["has_ticker"]].groupby(["year", "ticker"])["has_price"].any()
    out["tickers_never_priced"] = (~ever_priced).groupby(level="year").sum()
    return out


BROAD_TOP_N = 500
BROAD_CANDIDATE_POOL = 800  # CUSIPs ranked per date before ETFs/funds are removed (they rank high by value)
# From 3 Jan 2023 the SEC requires 13F values in dollars; before, in thousands.
# Summing across filers on one date mixes both around the switch unless normalized.
THIRTEEN_F_DOLLAR_VALUES_FROM = "2023-01-03"


def _latest_filings(filings: pd.DataFrame, date: pd.Timestamp, max_filing_age_days: int) -> pd.DataFrame:
    window = filings[(filings["filing_date"] <= date) & (filings["filing_date"] >= date - pd.Timedelta(days=max_filing_age_days))]
    return window.sort_values("filing_date").groupby("cik").tail(1)


def pit_broad_universe(
    history: pd.DataFrame,
    dates: Iterable,
    top_n: int = BROAD_TOP_N,
    max_filing_age_days: int = PIT_MAX_FILING_AGE_DAYS,
    holdings_loader: Optional[Callable] = None,
    resolve: Optional[Callable[[list[str]], dict]] = None,
    candidate_pool: int = BROAD_CANDIDATE_POOL,
) -> pd.DataFrame:
    """Broad point-in-time universe: the `top_n` stocks by total value across
    the filers' latest FULL 13F holdings, per date.

    Why: the selection universe (`pit_universe`, top 30 + movers per filer)
    gives ~100 names a date -- too few for a cross-sectional test to detect a
    small edge. The full holdings of the index giants cover the whole US
    market, so their summed value is a point-in-time "largest US stocks" list
    with the same availability rule (SEC filing date, 200-day staleness).

    - Filings are indexed from `history` (built by `build_universe_history`);
      each accession's holdings come from the per-accession parquet cache
      (`holdings_13f.load_holdings`), so nothing already cached is re-downloaded.
    - ETFs/funds are excluded via the CUSIP resolution's security type;
      unresolved CUSIPs stay as members with ticker=None, counted by coverage.
    - `institution_count` = filers holding it; `institutional_direction_score`
      = the live aggregation over the filers' selected rows (0 if none).
    """
    if history is None or history.empty:
        return pd.DataFrame(columns=MEMBERSHIP_COLUMNS + ["total_13f_value"])
    pace = resolve is None
    resolve = resolve or holdings_13f._default_resolver()
    loader = holdings_loader or holdings_13f.load_holdings
    history = history.assign(filing_date=pd.to_datetime(history["filing_date"]))
    filings = history[["cik", "accession", "filing_date", "report_period"]].drop_duplicates(["cik", "accession"])

    cache: dict[str, pd.DataFrame] = {}

    def holdings(row) -> pd.DataFrame:
        if row.accession not in cache:
            filed = f"{row.filing_date:%Y-%m-%d}"
            filing = holdings_13f.Filing13F(int(row.cik), row.accession, filed, str(row.report_period))
            frame = loader(filing)[["cusip", "value"]].copy()
            if filed < THIRTEEN_F_DOLLAR_VALUES_FROM:
                frame["value"] = frame["value"] * 1000.0
            cache[row.accession] = frame.assign(cik=int(row.cik))
        return cache[row.accession]

    candidates: dict[pd.Timestamp, pd.DataFrame] = {}
    latest_by_date: dict[pd.Timestamp, set] = {}
    for date in sorted(pd.Timestamp(d) for d in dates):
        latest = _latest_filings(filings, date, max_filing_age_days)
        if latest.empty:
            continue
        frames = []
        for row in latest.itertuples(index=False):
            try:
                frames.append(holdings(row))
            except Exception as exc:  # a filing that can't be loaded drops that filer for the date, not the run
                logger.warning("pit_broad: holdings for %s unavailable (%s)", row.accession, exc)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        agg = combined.groupby("cusip").agg(total_13f_value=("value", "sum"), institution_count=("cik", "nunique"))
        candidates[date] = agg.nlargest(candidate_pool, "total_13f_value").reset_index()
        latest_by_date[date] = set(latest["accession"])

    all_cusips = sorted({c for frame in candidates.values() for c in frame["cusip"]})
    resolutions = holdings_13f._paced_resolve(resolve, all_cusips, pace)
    has_ticker = history["ticker"].notna() & history["ticker"].astype(str).str.strip().ne("")

    rows = []
    for date, cand in candidates.items():
        info = [resolutions.get(c) or {} for c in cand["cusip"]]
        cand = cand.assign(
            ticker=[(i.get("ticker") or None) for i in info],
            excluded=[i.get("security_type") in holdings_13f.EXCLUDED_SECURITY_TYPES for i in info],
        )
        cand = cand[~cand["excluded"]].copy()
        cand["key"] = cand["ticker"].fillna("CUSIP:" + cand["cusip"])
        members = (
            cand.groupby("key")
            .agg(ticker=("ticker", "first"), cusip=("cusip", "first"),
                 total_13f_value=("total_13f_value", "sum"), institution_count=("institution_count", "max"))
            .nlargest(top_n, "total_13f_value")
        )
        selected = history[history["accession"].isin(latest_by_date[date]) & has_ticker]
        direction = {str(t).upper(): inst_universe.aggregate_direction(g["view_direction"])[0] for t, g in selected.groupby("ticker")}
        for m in members.itertuples(index=False):
            ticker = str(m.ticker).upper() if isinstance(m.ticker, str) and m.ticker else None
            rows.append({
                "date": date, "ticker": ticker, "cusip": m.cusip,
                "institution_count": int(m.institution_count),
                "institutional_direction_score": direction.get(ticker, 0.0) if ticker else None,
                "total_13f_value": float(m.total_13f_value),
            })
    return pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS + ["total_13f_value"])


ACTIVE_FEATURE_COLUMNS = ["active_new_or_add", "active_weight_max"]


def active_manager_features(
    keys: pd.DataFrame, position_history: pd.DataFrame, max_filing_age_days: int = PIT_MAX_FILING_AGE_DAYS
) -> pd.DataFrame:
    """Best-ideas 13F features per (date, ticker), from concentrated active managers.

      active_new_or_add  1 if any listed manager's latest filing (filing_date <= date)
                         opened the position or added > 10% flow-adjusted (status NEW/ADDED)
      active_weight_max  largest portfolio weight among those managers (0 if none holds it)

    NaN on dates where no listed manager has a filing in the staleness window
    ("no information", not "not held").
    """
    out = keys[["date", "ticker"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    if out.empty or position_history is None or position_history.empty:
        return out.assign(**{c: np.nan for c in ACTIVE_FEATURE_COLUMNS})
    ph = position_history.assign(filing_date=pd.to_datetime(position_history["filing_date"]))
    ph = ph[ph["ticker"].notna()].assign(ticker=lambda f: f["ticker"].astype(str).str.upper())
    filings = ph[["cik", "accession", "filing_date"]].drop_duplicates()

    parts = []
    for date in out["date"].unique():
        latest = _latest_filings(filings, pd.Timestamp(date), max_filing_age_days)
        if latest.empty:
            continue
        rows = ph[ph["accession"].isin(set(latest["accession"]))]
        per_ticker = rows.groupby("ticker").agg(
            active_new_or_add=("status", lambda s: float(s.isin(["NEW", "ADDED"]).any())),
            active_weight_max=("weight", "max"),
        ).reset_index()
        keys_today = out.loc[out["date"] == date, ["date", "ticker"]].drop_duplicates()
        merged = keys_today.merge(per_ticker, on="ticker", how="left")
        parts.append(merged.fillna({"active_new_or_add": 0.0, "active_weight_max": 0.0}))
    if not parts:
        return out.assign(**{c: np.nan for c in ACTIVE_FEATURE_COLUMNS})
    return out.merge(pd.concat(parts, ignore_index=True), on=["date", "ticker"], how="left")


def add_step13_features(
    panel: pd.DataFrame,
    active_path,
    insider_source: str = "bulk",
    refresh_active: bool = False,
    max_filing_age_days: int = PIT_MAX_FILING_AGE_DAYS,
) -> pd.DataFrame:
    """Insider (Form 4) and best-ideas (active managers' 13F) features, merged by (date, ticker).

    Computed after the panel so they can be added to a saved panel without
    re-downloading prices; both are point-in-time by SEC filing date. The
    active managers' position history is cached at `active_path`.
    """
    import sys

    import insider_form4

    panel = panel.drop(columns=[c for c in insider_form4.FEATURE_COLUMNS + ACTIVE_FEATURE_COLUMNS if c in panel.columns])
    keys = panel[["date", "ticker"] + (["log_market_cap"] if "log_market_cap" in panel.columns else [])]
    first, last = panel["date"].min(), panel["date"].max()

    print("STEP 13: Form 4 insider transactions ...", file=sys.stderr, flush=True)
    transactions, ciks, coverage_end = insider_form4.load_transactions(
        panel["ticker"].unique(), f"{first - pd.Timedelta(days=insider_form4.WINDOW_DAYS):%Y-%m-%d}", f"{last:%Y-%m-%d}",
        source=insider_source,
    )
    insider = insider_form4.insider_features(keys, transactions, ciks, coverage_end)
    print(f"  {len(transactions)} officer/director P/S rows; coverage to {coverage_end}", file=sys.stderr)

    if refresh_active or not active_path.exists():
        print("STEP 13: active managers' full 13F books ...", file=sys.stderr, flush=True)
        positions, summaries = holdings_13f.build_position_history(start_year=first.year - 1)
        for summary in summaries:
            print(f"  {summary}", file=sys.stderr)
        active_path.parent.mkdir(parents=True, exist_ok=True)
        positions.to_parquet(active_path, index=False)
    active = active_manager_features(keys, pd.read_parquet(active_path), max_filing_age_days)

    return (panel.merge(insider, on=["date", "ticker"], how="left")
                 .merge(active, on=["date", "ticker"], how="left"))


def _score(technicals: dict, fundamentals: Optional[dict], horizon_days: int) -> Optional[float]:
    return horizon.compute_horizon_weighted_view(technicals or None, fundamentals, horizon_days)["score"]


def build_panel(
    tickers: Optional[Iterable[str]],
    price_loader: Callable[[str], pd.DataFrame],
    fundamentals_loader: Optional[Callable[[str], dict]],
    benchmark: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    label_days: int = DEFAULT_LABEL_DAYS,
    rebalance_every: int = DEFAULT_REBALANCE_EVERY,
    fundamental_lag_days: int = pit_features.DEFAULT_FUNDAMENTAL_LAG_DAYS,
    start=None,
    end=None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    universe: str = "current",
    universe_history: Optional[pd.DataFrame] = None,
    max_filing_age_days: int = PIT_MAX_FILING_AGE_DAYS,
    membership: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """`universe="current"`: every ticker on every date. `universe="pit"`:
    each ticker only on the dates it is a PIT member (`pit_universe`, or a
    precomputed `membership`); `universe="pit_broad"`: same, with the
    `pit_broad_universe` membership (required, it is expensive to build).
    `tickers`, if given, further restricts the members.
    """
    if universe not in UNIVERSES:
        raise ValueError(f"universe must be one of {UNIVERSES}, got {universe!r}")
    benchmark = benchmark.sort_index()
    calendar = benchmark.index
    dates = rebalance_dates(calendar, rebalance_every, start=start, end=end)
    bench_closes = benchmark["adj_close"]

    member_scores: Optional[dict[str, dict]] = None
    if universe != "current":
        if membership is None and universe == "pit_broad":
            raise ValueError("universe='pit_broad' needs a membership frame (dataset.pit_broad_universe)")
        if membership is None and universe_history is None:
            raise ValueError("universe='pit' needs universe_history (holdings_13f.build_universe_history)")
        if membership is None:
            membership = pit_universe(universe_history, dates, max_filing_age_days)
        membership = membership.dropna(subset=["ticker"])
        membership = membership[membership["date"].isin(dates)]
        member_scores = {
            t: dict(zip(g["date"], zip(g["institutional_direction_score"], g["institution_count"])))
            for t, g in membership.groupby("ticker")
        }
        wanted = {t.upper() for t in tickers} if tickers else None
        tickers = [t for t in sorted(member_scores) if wanted is None or t in wanted]
    tickers = list(dict.fromkeys(t.upper() for t in tickers))

    rows: list[dict] = []
    for i, ticker in enumerate(tickers, 1):
        if progress:
            progress(i, len(tickers), ticker)
        ticker_dates = dates if member_scores is None else [d for d in dates if d in member_scores[ticker]]
        if not ticker_dates:
            continue
        try:
            prices = price_loader(ticker).sort_index()
        except Exception as exc:  # one bad ticker never stops the run (PIT: counted by universe_coverage)
            logger.warning("Skipping %s: price load failed (%s)", ticker, exc)
            continue
        if prices.empty:
            continue
        raw_fund = {}
        if fundamentals_loader is not None:
            try:
                raw_fund = fundamentals_loader(ticker) or {}
            except Exception as exc:
                logger.warning("No fundamentals for %s (%s)", ticker, exc)
        stock_closes = labels.aligned_closes(prices, calendar)
        price_now = float(prices["adj_close"].iloc[-1])

        for date in ticker_dates:
            tech = pit_features.pit_technicals(prices, date, benchmark)
            if not tech:
                continue
            fund = pit_features.pit_fundamentals(
                raw_fund, date, price_at_as_of=tech.get("adj_close"), price_now=price_now, lag_days=fundamental_lag_days
            ) if raw_fund else {}
            has_fund = any(fund.get(f) is not None for fs in horizon.GROUP_FIELDS.values() for f in fs if f in fund)
            fwd_ret, fwd_excess = labels.forward_returns(stock_closes, bench_closes, date, label_days)
            score_full = _score(tech, fund if has_fund else None, horizon_days)
            vol60 = tech.get("volatility_60d")
            row = {
                "date": date,
                "ticker": ticker,
                **{k: v for k, v in tech.items() if k != "adj_close"},
                **{k: v for k, v in fund.items() if k not in ("market_cap",)},
                "has_fundamentals": has_fund,
                # Explicit size feature: without it a model can use institution_count as a hidden size proxy.
                "log_market_cap": float(np.log(fund["market_cap"])) if (fund.get("market_cap") or 0) > 0 else None,
                "score_full": score_full,
                "score_technical": _score(tech, None, horizon_days),
                "label": screener.label_for(score_full),
                "low_volatility": -vol60 if vol60 is not None else None,
                "fwd_return": fwd_ret,
                "fwd_excess": fwd_excess,
            }
            if member_scores is not None:
                inst, count = member_scores[ticker][date]
                inst = 0.0 if inst is None or pd.isna(inst) else float(inst)  # no directional row = MENTIONED (0), as live
                # Same formula as screener._row_for: the production rank_score, and its label.
                row["institutional_direction_score"] = inst
                row["institution_count"] = count
                row["score_rank"] = None if score_full is None else score_full + screener.INSTITUTIONAL_TILT * inst
                row["label"] = screener.label_for(row["score_rank"])
            rows.append(row)

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel
