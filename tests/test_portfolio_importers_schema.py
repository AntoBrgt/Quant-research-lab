"""Canonical schema + reconstruct_positions -- pure Python, no network, no broker."""

from datetime import datetime

import pytest

from portfolio_importers.schema import CanonicalPosition, CanonicalTransaction, reconstruct_positions, to_simple_portfolio


def _tx(side, qty, price, date="2024-01-01", instrument_id="US123", fees=0.0, tax=0.0, currency="EUR", tx_id=None):
    return CanonicalTransaction(
        transaction_id=tx_id or f"tx-{date}-{side}-{qty}",
        date=datetime.fromisoformat(date),
        instrument_id=instrument_id,
        identity_type="ISIN",
        name="Test Co",
        symbol=instrument_id,
        asset_class="STOCK",
        side=side,
        quantity=qty,
        price=price,
        currency=currency,
        fees=fees,
        tax=tax,
        amount=None,
        broker="test",
    )


def test_single_buy_creates_a_position():
    positions = reconstruct_positions([_tx("BUY", 10, 100.0)])
    assert len(positions) == 1
    assert positions[0].quantity == 10
    assert positions[0].average_cost == 100.0
    assert positions[0].total_invested == 1000.0


def test_fractional_shares_are_supported():
    positions = reconstruct_positions([_tx("BUY", 0.025052, 97.552, tx_id="t1")])
    assert positions[0].quantity == pytest.approx(0.025052)


def test_multiple_buys_average_cost_is_quantity_weighted():
    transactions = [
        _tx("BUY", 1.0, 100.0, date="2024-01-01", tx_id="t1"),
        _tx("BUY", 1.0, 200.0, date="2024-01-02", tx_id="t2"),
    ]
    positions = reconstruct_positions(transactions)
    assert positions[0].quantity == 2.0
    assert positions[0].average_cost == 150.0  # (1*100 + 1*200) / 2


def test_buy_then_partial_sell_reconstruction():
    transactions = [
        _tx("BUY", 10.0, 100.0, date="2024-01-01", tx_id="t1"),
        _tx("SELL", 4.0, 999.0, date="2024-01-02", tx_id="t2"),  # sell price doesn't affect remaining cost basis
    ]
    positions = reconstruct_positions(transactions)
    assert positions[0].quantity == 6.0
    assert positions[0].average_cost == 100.0  # unchanged by the sell
    assert positions[0].total_invested == 600.0


def test_transactions_are_processed_in_date_order_regardless_of_input_order():
    # Fed in reverse chronological order -- reconstruction must still sort by date.
    transactions = [
        _tx("SELL", 1.0, 999.0, date="2024-01-02", tx_id="t2"),
        _tx("BUY", 2.0, 100.0, date="2024-01-01", tx_id="t1"),
    ]
    positions = reconstruct_positions(transactions)
    assert positions[0].quantity == 1.0
    assert positions[0].average_cost == 100.0


def test_zero_quantity_after_full_sell_is_not_a_current_position():
    transactions = [
        _tx("BUY", 5.0, 100.0, date="2024-01-01", tx_id="t1"),
        _tx("SELL", 5.0, 120.0, date="2024-01-02", tx_id="t2"),
    ]
    assert reconstruct_positions(transactions) == []


def test_fees_and_tax_accumulate_across_all_transactions_for_the_instrument():
    transactions = [
        _tx("BUY", 1.0, 100.0, date="2024-01-01", tx_id="t1", fees=1.0, tax=0.0),
        _tx("BUY", 1.0, 100.0, date="2024-01-02", tx_id="t2", fees=1.0, tax=0.5),
    ]
    positions = reconstruct_positions(transactions)
    assert positions[0].total_fees == pytest.approx(2.5)  # 1.0 + 1.0 + 0.5


def test_dividend_and_cash_sides_do_not_affect_quantity():
    transactions = [
        _tx("BUY", 1.0, 100.0, date="2024-01-01", tx_id="t1"),
        CanonicalTransaction(
            transaction_id="t2", date=datetime(2024, 1, 3), instrument_id="US123", identity_type="ISIN",
            name="Test Co", symbol="US123", asset_class="STOCK", side="DIVIDEND",
            quantity=None, price=None, currency="EUR", fees=0.0, tax=0.0, amount=1.5, broker="test",
        ),
    ]
    positions = reconstruct_positions(transactions)
    assert positions[0].quantity == 1.0


def test_other_side_transactions_never_affect_reconstruction():
    # e.g. Trade Republic's MIGRATION rows, mapped to OTHER by the adapter.
    transactions = [
        _tx("BUY", 3.0, 100.0, date="2024-01-01", tx_id="t1"),
        CanonicalTransaction(
            transaction_id="t2", date=datetime(2024, 6, 1), instrument_id="US123", identity_type="ISIN",
            name="Test Co", symbol="US123", asset_class="STOCK", side="OTHER",
            quantity=999.0, price=999.0, currency="EUR", fees=0.0, tax=0.0, amount=None, broker="test",
        ),
    ]
    positions = reconstruct_positions(transactions)
    assert positions[0].quantity == 3.0  # untouched by the OTHER row despite its large quantity/price


def test_multiple_instruments_reconstructed_independently():
    transactions = [
        _tx("BUY", 1.0, 100.0, date="2024-01-01", tx_id="t1", instrument_id="US123"),
        _tx("BUY", 2.0, 50.0, date="2024-01-01", tx_id="t2", instrument_id="FR999", currency="EUR"),
    ]
    positions = {p.instrument_id: p for p in reconstruct_positions(transactions)}
    assert positions["US123"].quantity == 1.0
    assert positions["FR999"].quantity == 2.0


def test_multiple_currencies_are_preserved_per_position():
    transactions = [
        _tx("BUY", 1.0, 100.0, tx_id="t1", instrument_id="US123", currency="USD"),
        _tx("BUY", 1.0, 100.0, tx_id="t2", instrument_id="FR999", currency="EUR"),
    ]
    positions = {p.instrument_id: p for p in reconstruct_positions(transactions)}
    assert positions["US123"].currency == "USD"
    assert positions["FR999"].currency == "EUR"


def test_negative_quantity_is_rejected_at_the_schema_level():
    with pytest.raises(ValueError):
        _tx("BUY", -1.0, 100.0)  # direction belongs in `side`, quantity must be non-negative


def test_trade_side_without_instrument_id_raises():
    tx = CanonicalTransaction(
        transaction_id="t1", date=datetime(2024, 1, 1), instrument_id=None, identity_type=None,
        name=None, symbol=None, asset_class=None, side="BUY",
        quantity=1.0, price=100.0, currency="EUR", fees=0.0, tax=0.0, amount=None, broker="test",
    )
    with pytest.raises(ValueError):
        reconstruct_positions([tx])


def test_to_simple_portfolio_bridges_to_existing_schema():
    position = CanonicalPosition(
        instrument_id="US02079K3059", name="Alphabet", symbol="GOOGL", asset_class="STOCK",
        quantity=2.0, average_cost=150.0, total_invested=300.0, total_fees=2.0, currency="EUR",
    )
    df = to_simple_portfolio([position])
    assert list(df.columns) == ["ticker", "quantity", "average_cost", "currency"]
    assert df.iloc[0]["ticker"] == "GOOGL"


def test_to_simple_portfolio_falls_back_to_instrument_id_when_no_symbol():
    position = CanonicalPosition(
        instrument_id="IE00B4L5Y983", name="Some Fund", symbol=None, asset_class="FUND",
        quantity=1.0, average_cost=100.0, total_invested=100.0, total_fees=0.0, currency="EUR",
    )
    df = to_simple_portfolio([position])
    assert df.iloc[0]["ticker"] == "IE00B4L5Y983"  # honest passthrough, will fail ticker validation downstream
