"""Point-in-time features: what the app *would have seen* on a past date.

Two rules keep this free of look-ahead bias:

1. **Prices** -- every technical is computed through `market_features.py`'s
   `as_of`-gated functions (the same ones `technicals.py` uses), so only rows
   with `date <= as_of` are ever read. A ticker whose last price is more than
   `MAX_STALE_DAYS` before `as_of` (delisted / halted) gets no features rather
   than a frozen, stale snapshot.
2. **Fundamentals** -- a statement for a period ending on `period_end` is
   only usable from `period_end + lag_days` (default 90: the 10-K deadline
   for large accelerated filers is 60 days, plus slack for smaller filers).
   Using it on `period_end` itself -- what a naive backtest does -- leaks
   ~2-3 months of future information into every fundamental feature.

The output dicts use exactly the field names `horizon.GROUP_FIELDS` reads, so
the baseline score is `horizon.compute_horizon_weighted_view` itself -- the
scoring code under test is the production code, not a re-implementation.

Known, documented limitations (see README STEP 11):
- yfinance statements are *as currently restated*, not as originally filed.
- `forward_pe` (analyst estimates) cannot be reconstructed historically ->
  always None here, so the valuation group runs on `fcf_yield` alone.
- Historical market cap is approximated as today's market cap scaled by the
  adjusted-price ratio (ignores buybacks/issuance; dividends slightly bias it).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

import fundamentals as fund
import market_features as mf

DEFAULT_FUNDAMENTAL_LAG_DAYS = 90
MAX_STALE_DAYS = 7

RETURN_WINDOWS = {"20d": 20, "60d": 60, "252d": 252}


def _last_date(prices: pd.DataFrame, as_of: pd.Timestamp) -> Optional[pd.Timestamp]:
    idx = prices.index[prices.index <= as_of]
    return idx.max() if len(idx) else None


def pit_technicals(prices: pd.DataFrame, as_of, benchmark: Optional[pd.DataFrame] = None) -> dict:
    """The technical fields `horizon.GROUP_FIELDS` uses, as of `as_of`.

    Plus `momentum_12_1` (return from t-252 to t-21, the classic
    cross-sectional momentum factor) as a reference signal -- if the
    production score can't beat this one line, its extra complexity is not
    earning its keep.
    """
    as_of = pd.Timestamp(as_of)
    prices = prices.sort_index()
    last = _last_date(prices, as_of)
    if last is None or (as_of - last).days > MAX_STALE_DAYS:
        return {}

    out: dict[str, Optional[float]] = {}
    for label, window in RETURN_WINDOWS.items():
        out[f"return_{label}"] = mf.compute_returns(prices, as_of, window=window)
    out["volatility_20d"] = mf.compute_volatility(prices, as_of, window=20)
    out["volatility_60d"] = mf.compute_volatility(prices, as_of, window=60)
    out["volume_ratio"] = mf.compute_volume_ratio(prices, as_of, window=21)
    out["price_vs_ma50"] = mf.compute_price_vs_ma(prices, as_of, window=50)
    out["price_vs_ma200"] = mf.compute_price_vs_ma(prices, as_of, window=200)

    if benchmark is not None and not benchmark.empty:
        for label, window in RETURN_WINDOWS.items():
            stock_ret = out[f"return_{label}"]
            bench_ret = mf.compute_returns(benchmark, as_of, window=window)
            out[f"relative_return_{label}"] = (
                stock_ret - bench_ret if (stock_ret is not None and bench_ret is not None) else None
            )

    closes = prices.loc[prices.index <= as_of, "adj_close"]
    out["momentum_12_1"] = float(closes.iloc[-22] / closes.iloc[-253] - 1) if len(closes) >= 253 else None
    out["adj_close"] = float(closes.iloc[-1])
    return out


def _available(series: dict, as_of: pd.Timestamp, lag_days: int) -> dict:
    """Keep only periods whose statement was public by `as_of`."""
    cutoff = as_of - pd.Timedelta(days=lag_days)
    kept = {}
    for period_end, value in (series or {}).items():
        try:
            if pd.Timestamp(period_end) <= cutoff:
                kept[period_end] = value
        except (ValueError, TypeError):
            continue
    return kept


def _latest_value(series: dict) -> Optional[float]:
    periods = fund._sorted_periods(series)
    return series[periods[-1]] if periods else None


def pit_fundamentals(
    raw: dict,
    as_of,
    price_at_as_of: Optional[float] = None,
    price_now: Optional[float] = None,
    lag_days: int = DEFAULT_FUNDAMENTAL_LAG_DAYS,
) -> dict:
    """The fundamental fields `horizon.GROUP_FIELDS` uses, as of `as_of`.

    `raw` is `YFinanceFundamentalsProvider.get_raw_fundamentals()` output.
    Reuses `fundamentals.py`'s own row lookups / ratio / growth helpers, so a
    metric means the same thing here as on the Company Research page -- the
    only difference is which periods are allowed in.
    """
    as_of = pd.Timestamp(as_of)
    raw = raw or {}
    income = raw.get("income_stmt") or {}
    cashflow = raw.get("cashflow") or {}
    balance = raw.get("balance_sheet") or {}
    info = raw.get("info") or {}

    def avail(statement: dict, *rows: str) -> dict:
        return _available(fund._first_present(statement, *rows), as_of, lag_days)

    revenue = avail(income, "Total Revenue", "TotalRevenue")
    net_income = avail(income, "Net Income", "NetIncome")
    eps = avail(income, "Diluted EPS", "DilutedEPS", "Basic EPS")
    op_cf = avail(cashflow, "Operating Cash Flow", "Total Cash From Operating Activities")
    capex = avail(cashflow, "Capital Expenditure", "CapitalExpenditure")
    fcf = avail(cashflow, "Free Cash Flow", "FreeCashFlow") or fund._raw_sum(op_cf, capex)
    equity = avail(balance, "Stockholders Equity", "Total Stockholder Equity", "TotalStockholderEquity")
    cash = avail(balance, "Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash")
    debt = avail(balance, "Total Debt", "TotalDebt")

    out: dict[str, Optional[float]] = {
        "revenue_growth": _latest_value(fund._raw_growth(revenue)),
        "eps_growth": _latest_value(fund._raw_growth(eps)),
        "net_margin": _latest_value(fund._raw_ratio(net_income, revenue)),
        "roe": _latest_value(fund._raw_ratio(net_income, equity)),
        "fcf_margin": _latest_value(fund._raw_ratio(fcf, revenue)),
        "fcf_growth": _latest_value(fund._raw_growth(fcf)),
        "forward_pe": None,  # analyst estimates: not reconstructable point-in-time
        "market_cap": None,
        "net_debt": None,
        "fcf_yield": None,
    }

    market_cap_now = info.get("marketCap")
    if market_cap_now and price_at_as_of and price_now:
        out["market_cap"] = market_cap_now * price_at_as_of / price_now
    latest_fcf = _latest_value(fcf)
    if out["market_cap"] and latest_fcf is not None:
        out["fcf_yield"] = latest_fcf / out["market_cap"]
    latest_debt, latest_cash = _latest_value(debt), _latest_value(cash)
    if latest_debt is not None and latest_cash is not None:
        out["net_debt"] = latest_debt - latest_cash

    periods = fund._sorted_periods(revenue)
    out["fundamentals_period_end"] = periods[-1] if periods else None
    return out
