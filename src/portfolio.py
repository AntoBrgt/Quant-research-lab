"""Portfolio loading, validation, and position-level math.

All of this is user-specific and deterministic Python -- no LLM, no shared
cache. `analyze_portfolio`-style callers pass a portfolio in as an argument
(never a module-level global), so the same functions work for any number of
portfolios without any per-user code path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

import price_provider

REQUIRED_COLUMNS = ["ticker", "quantity", "average_cost"]
TICKER_PATTERN = re.compile(r"^[A-Z]{1,6}([.\-][A-Z]{1,3})?$")


def load_portfolio_csv(path: Path) -> pd.DataFrame:
    """Load a raw portfolio CSV (ticker, quantity, average_cost[, currency])."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV is missing required columns: {sorted(missing)}")
    if "currency" not in df.columns:
        df["currency"] = "USD"
    return df


def validate_portfolio(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate rows; return (clean_rows, human-readable error list).

    Invalid rows are dropped from the returned frame but every problem is
    reported, so nothing fails silently.
    """
    errors: list[str] = []
    df = df.copy()

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    valid_mask = pd.Series(True, index=df.index)

    missing_ticker = df["ticker"].isin(["", "NAN", "NONE"])
    if missing_ticker.any():
        errors.append(f"{missing_ticker.sum()} row(s) have a missing ticker")
        valid_mask &= ~missing_ticker

    invalid_ticker = ~df["ticker"].str.match(TICKER_PATTERN) & ~missing_ticker
    if invalid_ticker.any():
        errors.append(f"Invalid ticker symbol(s): {sorted(df.loc[invalid_ticker, 'ticker'].unique().tolist())}")
        valid_mask &= ~invalid_ticker

    quantity = pd.to_numeric(df["quantity"], errors="coerce")
    non_numeric_qty = quantity.isna()
    if non_numeric_qty.any():
        errors.append(f"{non_numeric_qty.sum()} row(s) have a non-numeric quantity")
    negative_qty = quantity < 0
    if negative_qty.fillna(False).any():
        errors.append(f"Negative quantity for: {sorted(df.loc[negative_qty.fillna(False), 'ticker'].tolist())}")
    valid_mask &= ~non_numeric_qty & ~negative_qty.fillna(True)
    df["quantity"] = quantity

    avg_cost = pd.to_numeric(df["average_cost"], errors="coerce")
    missing_cost = avg_cost.isna() | (avg_cost <= 0)
    if missing_cost.any():
        errors.append(f"Missing/invalid average_cost for: {sorted(df.loc[missing_cost, 'ticker'].tolist())}")
    valid_mask &= ~missing_cost
    df["average_cost"] = avg_cost

    duplicate_ticker = df["ticker"].duplicated(keep=False) & valid_mask
    if duplicate_ticker.any():
        errors.append(f"Duplicate ticker rows for: {sorted(df.loc[duplicate_ticker, 'ticker'].unique().tolist())}")
        valid_mask &= ~duplicate_ticker

    clean = df.loc[valid_mask].reset_index(drop=True)
    return clean, errors


def _sector_for(ticker: str) -> Optional[str]:
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
        return info.get("sector")
    except Exception:
        return None


def enrich_positions(
    df: pd.DataFrame,
    price_prov: Optional[price_provider.PriceProvider] = None,
    include_sector: bool = True,
) -> pd.DataFrame:
    """Add current_price, market_value, weight, unrealized P/L, and concentration."""
    df = df.copy()

    current_prices = []
    for ticker in df["ticker"]:
        try:
            history = price_provider.get_price_history(ticker, provider=price_prov)
            current_prices.append(float(history["adj_close"].iloc[-1]) if not history.empty else None)
        except Exception:
            current_prices.append(None)
    df["current_price"] = current_prices

    df["market_value"] = df["quantity"] * df["current_price"]
    total_value = df["market_value"].sum(skipna=True)
    df["portfolio_weight"] = df["market_value"] / total_value if total_value else None

    df["cost_basis"] = df["quantity"] * df["average_cost"]
    df["unrealized_pl"] = df["market_value"] - df["cost_basis"]
    df["unrealized_pl_pct"] = (df["unrealized_pl"] / df["cost_basis"]).where(df["cost_basis"] > 0)

    if include_sector:
        df["sector"] = df["ticker"].map(lambda t: _sector_for(t) or "unavailable")
    else:
        df["sector"] = "unavailable"

    return df.sort_values("market_value", ascending=False, na_position="last").reset_index(drop=True)


def portfolio_concentration(enriched: pd.DataFrame, top_n: int = 3) -> dict:
    """Herfindahl-Hirschman index (0-1) and top-N weight, both concentration measures."""
    weights = enriched["portfolio_weight"].dropna()
    if weights.empty:
        return {"hhi": None, "top_n_weight": None, "top_n": top_n}
    return {
        "hhi": float((weights ** 2).sum()),
        "top_n_weight": float(weights.sort_values(ascending=False).head(top_n).sum()),
        "top_n": top_n,
    }


def sector_exposure(enriched: pd.DataFrame) -> dict:
    """Weight by sector, or an explicit note if sector data wasn't available."""
    if (enriched["sector"] == "unavailable").all():
        return {"available": False, "exposure": {}}
    grouped = enriched.groupby("sector")["portfolio_weight"].sum(min_count=1)
    return {"available": True, "exposure": grouped.dropna().to_dict()}
