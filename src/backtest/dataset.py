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

import pandas as pd

import horizon
import screener
from backtest import labels, pit_features
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
) -> pd.DataFrame:
    """`universe="current"`: every ticker on every date. `universe="pit"`:
    each ticker only on the dates it is a PIT member (`pit_universe`);
    `tickers`, if given, further restricts the members.
    """
    if universe not in ("current", "pit"):
        raise ValueError(f"universe must be 'current' or 'pit', got {universe!r}")
    benchmark = benchmark.sort_index()
    calendar = benchmark.index
    dates = rebalance_dates(calendar, rebalance_every, start=start, end=end)
    bench_closes = benchmark["adj_close"]

    member_scores: Optional[dict[str, dict]] = None
    if universe == "pit":
        if universe_history is None:
            raise ValueError("universe='pit' needs universe_history (holdings_13f.build_universe_history)")
        membership = pit_universe(universe_history, dates, max_filing_age_days).dropna(subset=["ticker"])
        member_scores = {t: dict(zip(g["date"], g["institutional_direction_score"])) for t, g in membership.groupby("ticker")}
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
                "score_full": score_full,
                "score_technical": _score(tech, None, horizon_days),
                "label": screener.label_for(score_full),
                "low_volatility": -vol60 if vol60 is not None else None,
                "fwd_return": fwd_ret,
                "fwd_excess": fwd_excess,
            }
            if member_scores is not None:
                inst = member_scores[ticker][date] or 0.0
                # Same formula as screener._row_for: the production rank_score, and its label.
                row["institutional_direction_score"] = inst
                row["score_rank"] = None if score_full is None else score_full + screener.INSTITUTIONAL_TILT * inst
                row["label"] = screener.label_for(row["score_rank"])
            rows.append(row)

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel
