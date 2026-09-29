"""PHASE 2 signal generators (PREREGISTRATION_PHASE2.md). Each returns events:
signal_date, ticker, strength -- the day the information became public and how
strong it was. Nothing here reads a price or a filing dated after signal_date.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["signal_date", "ticker", "strength"]

H1_WINDOW_DAYS = 30
H1_MIN_BUYERS = 3
H1_COOLDOWN_DAYS = 90
H2_PERCENTILE = 0.90
H2_LOOKBACK_DAYS = 365
H2_MIN_HISTORY = 100
H3_MA_DAYS = 200


def insider_cluster_events(transactions: pd.DataFrame, cik_to_ticker: dict[int, str],
                           window_days: int = H1_WINDOW_DAYS, min_buyers: int = H1_MIN_BUYERS,
                           cooldown_days: int = H1_COOLDOWN_DAYS) -> pd.DataFrame:
    """H1: on filing date D, distinct officer/director open-market buyers filed in (D - window, D] >= min_buyers,
    and no H1 signal for that issuer in the previous `cooldown_days`.

    strength = buyers + (purchase value in the window) / 1e10: more buyers first, then bigger purchases.
    Transactions are the `insider_form4` table (already officer/director P/S only, original Form 4s).
    """
    tx = transactions[transactions["trans_code"] == "P"].copy()
    if tx.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    tx["filing_date"] = pd.to_datetime(tx["filing_date"])
    tx["issuer_cik"] = pd.to_numeric(tx["issuer_cik"], errors="coerce")
    tx = tx.dropna(subset=["filing_date", "issuer_cik", "owner_cik"])
    values = tx.drop_duplicates(subset=["accession", "trans_date", "shares", "price"])
    window = pd.Timedelta(days=window_days)
    cooldown = pd.Timedelta(days=cooldown_days)
    rows = []
    for cik, g in tx.groupby("issuer_cik"):
        ticker = cik_to_ticker.get(int(cik))
        if ticker is None:
            continue
        v = values[values["issuer_cik"] == cik]
        last_signal: Optional[pd.Timestamp] = None
        for d in sorted(g["filing_date"].unique()):
            d = pd.Timestamp(d)
            in_window = g[(g["filing_date"] > d - window) & (g["filing_date"] <= d)]
            buyers = in_window["owner_cik"].nunique()
            if buyers < min_buyers:
                continue
            if last_signal is not None and d - last_signal <= cooldown:
                continue
            value = v[(v["filing_date"] > d - window) & (v["filing_date"] <= d)]["value"].sum()
            rows.append({"signal_date": d, "ticker": ticker,
                         "strength": float(buyers) + float(np.nan_to_num(value)) / 1e10})
            last_signal = d
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def earnings_surprise_events(sue_by_ticker: dict[str, pd.DataFrame], is_member,
                             percentile: float = H2_PERCENTILE, lookback_days: int = H2_LOOKBACK_DAYS,
                             min_history: int = H2_MIN_HISTORY) -> pd.DataFrame:
    """H2: a filing whose SUE >= the `percentile` of all member SUEs filed in [D - lookback, D).

    `sue_by_ticker[t]` is `sec_xbrl.sue_series` output (filed, sue); `is_member(date, ticker)`
    is the point-in-time universe test. The threshold uses only filings strictly before D.
    """
    frames = []
    for ticker, sue in sue_by_ticker.items():
        if sue is None or sue.empty:
            continue
        f = sue[["filed", "sue"]].dropna().copy()
        f["ticker"] = ticker
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    allsue = pd.concat(frames, ignore_index=True)
    allsue["filed"] = pd.to_datetime(allsue["filed"])
    allsue = allsue[[is_member(d, t) for d, t in zip(allsue["filed"], allsue["ticker"])]]
    allsue = allsue.sort_values(["filed", "ticker"]).reset_index(drop=True)
    filed = allsue["filed"].to_numpy()
    values = allsue["sue"].to_numpy(dtype=float)
    lookback = np.timedelta64(lookback_days, "D")
    rows = []
    for i in range(len(allsue)):
        d = filed[i]
        lo = np.searchsorted(filed, d - lookback, side="left")
        hi = np.searchsorted(filed, d, side="left")  # strictly before D
        if hi - lo < min_history:
            continue
        threshold = np.quantile(values[lo:hi], percentile)
        if values[i] >= threshold:
            rows.append({"signal_date": pd.Timestamp(d), "ticker": allsue["ticker"].iat[i], "strength": float(values[i])})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def cheap_quality_trend_events(members: pd.DataFrame, prices: dict[str, pd.DataFrame],
                               ma_days: int = H3_MA_DAYS) -> pd.DataFrame:
    """H3: STEP 15 candidate month-ends whose adjusted close is above its `ma_days` moving average.

    The average uses closes up to and including the month-end only. strength = drawdown.
    """
    cands = members[members["is_candidate"].fillna(False).astype(bool)]
    rows = []
    for ticker, g in cands.groupby("ticker"):
        if ticker not in prices:
            continue
        close = prices[ticker].sort_index()["adj_close"]
        close = close[~close.index.duplicated(keep="last")]
        ma = close.rolling(ma_days, min_periods=ma_days).mean()
        for r in g.itertuples(index=False):
            d = pd.Timestamp(r.date)
            c, m = close.asof(d), ma.asof(d)
            if pd.notna(c) and pd.notna(m) and c > m:
                rows.append({"signal_date": d, "ticker": ticker, "strength": float(r.drawdown)})
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
