"""Trade Republic transaction-export adapter.

Maps Trade Republic's raw CSV columns into `schema.CanonicalTransaction`
rows. This is the only file that knows Trade Republic's column names and
`type` vocabulary -- nothing outside this module (or `detect.py`, for format
detection) should reference them.

No network access, no live prices -- parsing works from the CSV alone.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from .schema import CanonicalTransaction, IdentityType

BROKER_NAME = "trade_republic"

# The minimal set of columns this adapter actually reads. Trade Republic's
# real export has more (account_type, category, description, counterparty_*,
# mcc_code, ...) -- those aren't needed for portfolio reconstruction and are
# left alone rather than modeled here.
REQUIRED_COLUMNS = ["datetime", "type", "name", "symbol", "shares", "price", "currency"]

# Trade Republic `type` -> canonical side. Anything not listed here becomes
# OTHER (preserved, not discarded, not treated as a trade) -- see
# schema.TransactionSide for why OTHER is safe by construction: only BUY/SELL
# affect position reconstruction.
#
# MIGRATION is deliberately OTHER, not BUY/SELL: it's a same-symbol, same-day
# internal re-booking (broker-side technical re-registration) that always
# nets to zero and carries no real cost information -- confirmed against a
# real export before writing this (every MIGRATION pair in that export summed
# to exactly zero shares per symbol).
TYPE_TO_SIDE = {
    "BUY": "BUY",
    "SELL": "SELL",
    "DIVIDEND": "DIVIDEND",
    "INTEREST_PAYMENT": "INTEREST",
    "CUSTOMER_INBOUND": "CASH_IN",
    "TRANSFER_INSTANT_INBOUND": "CASH_IN",
    "CUSTOMER_OUTBOUND_REQUEST": "CASH_OUT",
    "TRANSFER_INSTANT_OUTBOUND": "CASH_OUT",
}

_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def _resolve_identity(symbol, name) -> tuple[Optional[str], Optional[IdentityType]]:
    """instrument_id preference: ISIN > symbol/ticker > name > none.

    Trade Republic's `symbol` column holds an ISIN for stocks/funds (e.g.
    "US02079K3059") or a plain crypto ticker for crypto (e.g. "BTC") -- both
    are usable identifiers, just of different strength, which is why the
    identity type is reported alongside the id rather than silently unified.
    """
    symbol = symbol if isinstance(symbol, str) and symbol.strip() else None
    if symbol and _ISIN_PATTERN.match(symbol):
        return symbol, "ISIN"
    if symbol:
        return symbol, "SYMBOL"
    name = name if isinstance(name, str) and name.strip() else None
    if name:
        return name, "NAME"
    return None, None


def _to_float(value, field: str, row_id) -> Optional[float]:
    if pd.isna(value) or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"row {row_id}: {field}={value!r} is not a valid number")


class RejectedRow(dict):
    """A row that couldn't be parsed, plus why -- for reporting, not raising."""


def parse(df: pd.DataFrame) -> tuple[list[CanonicalTransaction], list[RejectedRow]]:
    """Parse a raw Trade Republic export into canonical transactions.

    Malformed rows (unparseable dates/numbers, a trade with no instrument
    identity) are collected in the returned rejected-rows list with a reason,
    rather than raising and discarding the whole import or being silently
    dropped -- see the module docstring on `raise` vs. `reject`.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Not a Trade Republic export: missing required columns {sorted(missing)}")

    transactions: list[CanonicalTransaction] = []
    rejected: list[RejectedRow] = []
    seen_transaction_ids: set[str] = set()

    for idx, row in df.iterrows():
        row_id = row.get("transaction_id") or f"row {idx}"
        try:
            raw_type = str(row["type"]).strip()
            side = TYPE_TO_SIDE.get(raw_type, "OTHER")

            date = pd.to_datetime(row["datetime"], errors="raise")

            quantity = _to_float(row.get("shares"), "shares", row_id)
            price = _to_float(row.get("price"), "price", row_id)
            fees = abs(_to_float(row.get("fee"), "fee", row_id) or 0.0)
            tax = abs(_to_float(row.get("tax"), "tax", row_id) or 0.0)
            amount = _to_float(row.get("amount"), "amount", row_id)

            instrument_id, identity_type = _resolve_identity(row.get("symbol"), row.get("name"))

            if side in ("BUY", "SELL") and instrument_id is None:
                raise ValueError(f"row {row_id}: {side} transaction has no symbol or name to identify the instrument")
            if side in ("BUY", "SELL") and (quantity is None or price is None):
                raise ValueError(f"row {row_id}: {side} transaction is missing shares or price")

            currency = row.get("currency")
            if not isinstance(currency, str) or not currency.strip():
                raise ValueError(f"row {row_id}: missing currency")

            transaction_id = str(row.get("transaction_id") or f"{BROKER_NAME}-row-{idx}")
            if transaction_id in seen_transaction_ids:
                raise ValueError(f"row {row_id}: duplicate transaction_id {transaction_id!r}")
            seen_transaction_ids.add(transaction_id)

            transactions.append(
                CanonicalTransaction(
                    transaction_id=transaction_id,
                    date=date,
                    instrument_id=instrument_id,
                    identity_type=identity_type,
                    name=row.get("name") if isinstance(row.get("name"), str) and row.get("name").strip() else None,
                    symbol=row.get("symbol") if isinstance(row.get("symbol"), str) and row.get("symbol").strip() else None,
                    asset_class=row.get("asset_class") if isinstance(row.get("asset_class"), str) and row.get("asset_class").strip() else None,
                    side=side,
                    quantity=abs(quantity) if quantity is not None else None,
                    price=price,
                    currency=currency.strip(),
                    fees=fees,
                    tax=tax,
                    amount=amount,
                    broker=BROKER_NAME,
                    raw_type=raw_type,
                )
            )
        except (ValueError, TypeError) as exc:
            rejected.append(RejectedRow(row_index=int(idx), reason=str(exc)))

    return transactions, rejected
