"""Canonical, broker-agnostic transaction and position schemas.

This is the one place transaction reconstruction happens. Every broker
adapter (e.g. `trade_republic.py`) only has to produce a list of
`CanonicalTransaction` -- `reconstruct_positions()` here is the single,
documented implementation of cost-basis accounting, shared by every adapter
so the methodology can't silently drift between brokers.

No network access and no market prices are used anywhere in this module --
reconstruction works from transaction data alone. Current-price enrichment is
a deliberately separate concern, handled by `src/portfolio.py`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Canonical transaction schema
# ---------------------------------------------------------------------------

# BUY/SELL are the only sides that change a position. Everything else is
# preserved (never discarded) but ignored by position reconstruction:
#   DIVIDEND, INTEREST  -- cash income, no quantity change
#   CASH_IN, CASH_OUT   -- external transfers into/out of the brokerage account
#   OTHER               -- anything a broker adapter can't confidently classify
#                          (e.g. Trade Republic's MIGRATION rows: a same-symbol,
#                          same-day internal re-booking, not a real trade)
TransactionSide = Literal["BUY", "SELL", "DIVIDEND", "INTEREST", "CASH_IN", "CASH_OUT", "OTHER"]

# How the instrument_id was derived, in the priority order instrument_id was
# chosen with -- kept so callers can judge how trustworthy the identity is.
IdentityType = Literal["ISIN", "SYMBOL", "NAME", "BROKER_ID"]


class CanonicalTransaction(BaseModel):
    """One transaction, independent of which broker it came from."""

    transaction_id: str
    date: datetime
    instrument_id: Optional[str] = None  # None only for pure cash movements (no instrument)
    identity_type: Optional[IdentityType] = None
    name: Optional[str] = None
    symbol: Optional[str] = None  # the broker's own identifier, preserved as-is (may be an ISIN)
    asset_class: Optional[str] = None
    side: TransactionSide
    quantity: Optional[float] = None  # always a non-negative magnitude; `side` carries direction
    price: Optional[float] = None
    currency: str
    fees: float = 0.0  # non-negative magnitude
    tax: float = 0.0  # non-negative magnitude
    amount: Optional[float] = None  # net cash flow of this transaction, sign preserved (broker convention)
    broker: str
    raw_type: Optional[str] = None  # the broker's own transaction-type string, for traceability

    @field_validator("quantity", "price", "fees", "tax")
    @classmethod
    def non_negative(cls, value):
        if value is not None and value < 0:
            raise ValueError("must be a non-negative magnitude; direction belongs in `side`, not the sign")
        return value


# ---------------------------------------------------------------------------
# Canonical position schema
# ---------------------------------------------------------------------------

class CanonicalPosition(BaseModel):
    """Current holding in one instrument, reconstructed from transactions."""

    instrument_id: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    asset_class: Optional[str] = None
    quantity: float
    average_cost: float  # cost basis per unit of the CURRENT quantity (see reconstruct_positions)
    total_invested: float  # quantity * average_cost -- cost basis of what's currently held
    total_fees: float  # sum of fees across every BUY/SELL for this instrument, ever (not reduced by sells)
    currency: str


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_positions(transactions: list[CanonicalTransaction]) -> list[CanonicalPosition]:
    """Aggregate BUY/SELL transactions into current positions.

    Cost-basis methodology: **moving-average cost** (not FIFO/LIFO lot
    tracking, not a tax accounting method -- this is for portfolio analytics
    only and makes no tax claim).

    - On BUY: new_average_cost = (old_qty*old_avg + buy_qty*buy_price) / new_qty;
      quantity increases.
    - On SELL: quantity decreases by the sold amount; average_cost of the
      remaining shares is UNCHANGED (selling doesn't change the cost basis of
      what's left, which is the defining property of the moving-average
      method, as opposed to FIFO/LIFO).

    Transactions are processed in `date` order per instrument, since the
    moving average is order-dependent. Only `side in (BUY, SELL)` affects
    quantity/cost; every other side is ignored here (by design -- e.g. a
    same-symbol wash re-booking classified as OTHER by an adapter never
    reaches this function's accounting). Positions whose quantity nets to
    (approximately) zero are dropped: they are not a current holding.
    """
    by_instrument: dict[str, list[CanonicalTransaction]] = {}
    for tx in transactions:
        if tx.side not in ("BUY", "SELL"):
            continue
        if tx.instrument_id is None:
            raise ValueError(f"Transaction {tx.transaction_id} has side={tx.side} but no instrument_id")
        by_instrument.setdefault(tx.instrument_id, []).append(tx)

    positions: list[CanonicalPosition] = []
    for instrument_id, txs in by_instrument.items():
        txs = sorted(txs, key=lambda t: t.date)

        quantity = 0.0
        average_cost = 0.0
        total_fees = 0.0

        for tx in txs:
            qty = tx.quantity or 0.0
            total_fees += tx.fees + tx.tax
            if tx.side == "BUY":
                new_quantity = quantity + qty
                average_cost = (quantity * average_cost + qty * (tx.price or 0.0)) / new_quantity if new_quantity else 0.0
                quantity = new_quantity
            else:  # SELL
                quantity -= qty

        if abs(quantity) < 1e-6:
            continue  # fully closed -- not a current holding

        last = txs[-1]
        positions.append(
            CanonicalPosition(
                instrument_id=instrument_id,
                name=last.name,
                symbol=last.symbol,
                asset_class=last.asset_class,
                quantity=quantity,
                average_cost=average_cost,
                total_invested=quantity * average_cost,
                total_fees=total_fees,
                currency=last.currency,
            )
        )

    return positions


# ---------------------------------------------------------------------------
# Backward-compat bridge to the existing simple portfolio.py schema
# ---------------------------------------------------------------------------

def to_simple_portfolio(positions: list[CanonicalPosition]) -> pd.DataFrame:
    """Bridge to `portfolio.py`'s existing `ticker, quantity, average_cost,
    currency` shape, so the existing validate_portfolio/enrich_positions/
    recommendation pipeline keeps working unchanged for imported portfolios.

    Uses `symbol` as the ticker candidate (falling back to `instrument_id`).
    This is deliberately NOT a security-master/ISIN-resolution step (out of
    scope for this layer, per its own docstring) -- an ISIN or other
    non-ticker symbol will simply fail `portfolio.py`'s own ticker
    validation, which is the honest outcome, not a silent guess.
    """
    return pd.DataFrame(
        [
            {
                "ticker": p.symbol or p.instrument_id,
                "quantity": p.quantity,
                "average_cost": p.average_cost,
                "currency": p.currency,
            }
            for p in positions
        ],
        columns=["ticker", "quantity", "average_cost", "currency"],
    )
