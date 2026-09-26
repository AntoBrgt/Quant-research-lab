"""Forward-return labels.

Entry is the close of the trading day AFTER the signal date (the signal uses
the signal date's close, so trading on that same close would be optimistic),
exit is `days` trading days after entry. The label is the stock's return
minus the benchmark's over the exact same entry/exit dates.

Both series are aligned to the benchmark's trading calendar; a stock with no
price at entry or exit (delisted, halted) gets no label -- never a
forward-filled one beyond `MAX_FILL_DAYS`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

MAX_FILL_DAYS = 5


def aligned_closes(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    closes = prices.sort_index()["adj_close"]
    closes = closes[~closes.index.duplicated(keep="last")]
    return closes.reindex(calendar.union(closes.index)).ffill(limit=MAX_FILL_DAYS).reindex(calendar)


def forward_returns(
    stock_closes: pd.Series, bench_closes: pd.Series, signal_date, days: int
) -> tuple[Optional[float], Optional[float]]:
    """(forward return, forward excess return vs benchmark), or (None, None).

    Both series must already be aligned to the same calendar (the
    benchmark's), e.g. via `aligned_closes`.
    """
    calendar = bench_closes.index
    pos = calendar.searchsorted(pd.Timestamp(signal_date))
    if pos >= len(calendar) or calendar[pos] != pd.Timestamp(signal_date):
        return None, None
    entry, exit_ = pos + 1, pos + 1 + days
    if exit_ >= len(calendar):
        return None, None
    s0, s1 = stock_closes.iloc[entry], stock_closes.iloc[exit_]
    b0, b1 = bench_closes.iloc[entry], bench_closes.iloc[exit_]
    if any(pd.isna(v) or v == 0 for v in (s0, s1, b0, b1)):
        return None, None
    stock_ret = float(s1 / s0 - 1)
    return stock_ret, stock_ret - float(b1 / b0 - 1)
