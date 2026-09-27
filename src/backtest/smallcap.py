"""STEP 14a -- small caps, where the big institutions can't easily trade.

Hypothesis: mispricing survives where large funds can't take meaningful
positions, so a ranking model has more room there than in the top 500.

Universe, point-in-time, monthly:
- every US common stock in the filers' latest FULL 13F books (index giants
  hold nearly every listed stock), ETFs / funds / ADRs excluded;
- market cap $200M-$2B = quoted price x XBRL shares outstanding (as filed by
  the date); 20-day average dollar volume > $1M; price > $3.

Pre-filter before any ticker lookup: the filers together own roughly
`OWNERSHIP_ASSUMPTION` of a typical US stock, so total 13F value / that share is a
rough market cap. Only CUSIPs whose rough cap is within a 4x-wide band around
$200M-$2B are resolved and priced -- the real filter comes after, on real data.

Survivorship: members the band keeps but yfinance can't price (delisted,
renamed) are counted per date. `survivorship_adjusted` then assumes the model
would have picked them at their universe share and that each lost 50% (or 100%)
over the 63-day label period.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest import dataset
from institutional_research import holdings_13f

logger = logging.getLogger(__name__)

MCAP_MIN, MCAP_MAX = 200e6, 2e9
MIN_DOLLAR_VOLUME = 1e6
MIN_PRICE = 3.0
OWNERSHIP_ASSUMPTION = 0.20
PREFILTER_MIN, PREFILTER_MAX = MCAP_MIN / 4, MCAP_MAX * 2
EXCLUDED_TYPES = set(holdings_13f.EXCLUDED_SECURITY_TYPES) | {"ADR", "GDR", "NY Reg Shrs", "Depositary Receipt"}
LABEL_DAYS = 63
COST_PER_SIDE = 0.0030
MAX_WEIGHT = 0.05
EMBARGO = 4
SENSITIVITY_LOSSES = (0.0, 0.50, 1.00)


def book_values(history: pd.DataFrame, dates, holdings_loader: Optional[Callable] = None,
                max_filing_age_days: int = dataset.PIT_MAX_FILING_AGE_DAYS) -> pd.DataFrame:
    """Per (date, cusip): total 13F value across the filers' latest full books, and how many hold it."""
    loader = holdings_loader or holdings_13f.load_holdings
    history = history.assign(filing_date=pd.to_datetime(history["filing_date"]))
    filings = history[["cik", "accession", "filing_date", "report_period"]].drop_duplicates(["cik", "accession"])
    cache: dict[str, pd.DataFrame] = {}
    frames = []
    for date in sorted(pd.Timestamp(d) for d in dates):
        latest = dataset._latest_filings(filings, date, max_filing_age_days)
        parts = []
        for row in latest.itertuples(index=False):
            if row.accession not in cache:
                filed = f"{row.filing_date:%Y-%m-%d}"
                try:
                    frame = loader(holdings_13f.Filing13F(int(row.cik), row.accession, filed, str(row.report_period)))
                except Exception as exc:
                    logger.warning("smallcap: holdings %s unavailable (%s)", row.accession, exc)
                    frame = pd.DataFrame(columns=["cusip", "value"])
                frame = frame[["cusip", "value"]].copy()
                if filed < dataset.THIRTEEN_F_DOLLAR_VALUES_FROM:
                    frame["value"] = frame["value"] * 1000.0
                cache[row.accession] = frame.assign(cik=int(row.cik))
            parts.append(cache[row.accession])
        if not parts:
            continue
        agg = pd.concat(parts).groupby("cusip").agg(total_13f_value=("value", "sum"), holders=("cik", "nunique")).reset_index()
        frames.append(agg.assign(date=date))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["cusip", "total_13f_value", "holders", "date"])
    out["rough_mcap"] = out["total_13f_value"] / OWNERSHIP_ASSUMPTION
    return out


def prefilter(books: pd.DataFrame) -> pd.DataFrame:
    return books[books["rough_mcap"].between(PREFILTER_MIN, PREFILTER_MAX)]


def dollar_volume_20d(prices: pd.DataFrame) -> pd.Series:
    """20-day average traded value. yfinance splits adjust close and volume in opposite directions, so close x volume is split-invariant."""
    return (prices["close_split_adj"] * prices["volume"]).rolling(20, min_periods=15).mean()


def apply_filters(members: pd.DataFrame) -> pd.Series:
    """True for rows that pass the STEP 14a filters (market cap, liquidity, price)."""
    return (members["market_cap"].between(MCAP_MIN, MCAP_MAX)
            & (members["dollar_volume_20d"] > MIN_DOLLAR_VOLUME)
            & (members["raw_price"] > MIN_PRICE))


def survivorship_adjusted(portfolio_returns: pd.Series, unpriced_share: pd.Series, loss_over_label: float,
                          period_days: int = 21, label_days: int = LABEL_DAYS) -> pd.Series:
    """Portfolio period returns if the model had also picked the unpriced members at their universe share.

    Each unpriced member is assumed to lose `loss_over_label` over the 63-day
    label period, i.e. (1 - loss) ** (period_days / label_days) - 1 per holding
    period (a -50% quarter is about -21% a month; -100% is -100%). The
    portfolio return becomes (1 - u) x r + u x that loss, u = unpriced share.
    """
    if loss_over_label >= 1.0:
        per_period = -1.0
    else:
        per_period = (1 - loss_over_label) ** (period_days / label_days) - 1
    u = unpriced_share.reindex(portfolio_returns.index).fillna(0.0).clip(0, 1)
    return (1 - u) * portfolio_returns + u * per_period


def ownership_breadth_change(members: pd.DataFrame, months: int = 3) -> pd.Series:
    """Holders now minus holders `months` month-ends earlier (13F ownership breadth change), per (date, ticker)."""
    frame = members[["date", "ticker", "holders"]].sort_values(["ticker", "date"])
    return frame.groupby("ticker")["holders"].diff(months).reindex(members.index)


def monthly_portfolio(panel: pd.DataFrame, score: str, next_returns: pd.Series, top_quantile: float = 0.10,
                      max_weight: float = MAX_WEIGHT, cost: float = COST_PER_SIDE) -> pd.DataFrame:
    """Long the top decile by `score` each month, equal weight (capped), costs on turnover.

    `next_returns` (aligned with `panel`) = return from the next open after the
    signal date to the next open after the following signal date. Returns one
    row per signal date: gross, turnover, net return, names.
    """
    rows, prev_w = [], pd.Series(dtype=float)
    for date, group in panel.dropna(subset=[score]).groupby("date"):
        n = max(int(math.floor(len(group) * top_quantile)), 1)
        top = group.nlargest(n, score)
        w = pd.Series(min(1.0 / n, max_weight), index=top["ticker"].to_numpy())
        r = next_returns.loc[top.index].to_numpy(dtype=float)
        r = np.where(np.isfinite(r), r, 0.0)
        gross = float((w.to_numpy() * r).sum())  # uninvested remainder (if capped) earns 0
        turnover = float(w.sub(prev_w, fill_value=0.0).abs().sum())
        rows.append({"date": date, "gross": gross, "turnover": turnover, "net": gross - turnover * cost, "names": n})
        prev_w = w * (1 + pd.Series(r, index=w.index))
        prev_w = prev_w / prev_w.sum() * w.sum() if prev_w.sum() > 0 else prev_w
    return pd.DataFrame(rows).set_index("date")


def annualized(returns: pd.Series, periods_per_year: float = 12) -> float:
    growth = float((1 + returns).prod())
    years = len(returns) / periods_per_year
    return growth ** (1 / years) - 1 if years > 0 and growth > 0 else -1.0
