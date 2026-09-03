"""Trade Republic adapter -- synthetic rows shaped like the real export, no network.

Column names/values mirror the real Trade Republic export schema without
hardcoding any of that export's actual transactions (per the task's
constraint) -- every row here is constructed fresh in the test.
"""

import pandas as pd
import pytest

from portfolio_importers import trade_republic


def _row(**overrides):
    base = {
        "datetime": "2024-10-09T09:15:36.798Z", "type": "BUY", "asset_class": "STOCK",
        "name": "Test Co", "symbol": "US02079K3059", "shares": "1.0", "price": "100.0",
        "amount": "-100.0", "fee": "-1.0", "tax": "", "currency": "EUR",
        "transaction_id": "tx-1",
    }
    base.update(overrides)
    return base


def test_buy_row_parses_to_canonical_buy():
    df = pd.DataFrame([_row()])
    transactions, rejected = trade_republic.parse(df)
    assert rejected == []
    assert len(transactions) == 1
    tx = transactions[0]
    assert tx.side == "BUY"
    assert tx.quantity == 1.0
    assert tx.price == 100.0
    assert tx.fees == 1.0  # sign stripped -- fees are a non-negative magnitude


def test_sell_row_parses_to_canonical_sell():
    df = pd.DataFrame([_row(type="SELL", shares="-1.0", amount="105.0", transaction_id="tx-2")])
    transactions, _ = trade_republic.parse(df)
    assert transactions[0].side == "SELL"
    assert transactions[0].quantity == 1.0  # magnitude, not the raw negative sign


def test_isin_symbol_resolves_as_isin_identity():
    df = pd.DataFrame([_row(symbol="US02079K3059")])
    transactions, _ = trade_republic.parse(df)
    assert transactions[0].instrument_id == "US02079K3059"
    assert transactions[0].identity_type == "ISIN"


def test_crypto_symbol_resolves_as_symbol_not_isin():
    df = pd.DataFrame([_row(symbol="BTC", asset_class="CRYPTO", transaction_id="tx-btc")])
    transactions, _ = trade_republic.parse(df)
    assert transactions[0].identity_type == "SYMBOL"


def test_cash_inbound_row_is_classified_as_cash_not_a_trade():
    df = pd.DataFrame([_row(
        type="CUSTOMER_INBOUND", asset_class="", name="ANTONIN BRENGETTO", symbol="",
        shares="", price="", amount="360.0", fee="", transaction_id="tx-cash",
    )])
    transactions, rejected = trade_republic.parse(df)
    assert rejected == []
    assert transactions[0].side == "CASH_IN"
    assert transactions[0].quantity is None


def test_migration_type_is_classified_as_other():
    df = pd.DataFrame([_row(type="MIGRATION", transaction_id="tx-mig")])
    transactions, _ = trade_republic.parse(df)
    assert transactions[0].side == "OTHER"


def test_unrecognized_type_is_preserved_as_other_not_dropped():
    df = pd.DataFrame([_row(
        type="SOME_NEW_TYPE_NOT_YET_SEEN", asset_class="", symbol="", shares="",
        price="", amount="-10.0", fee="", transaction_id="tx-unknown",
    )])
    transactions, rejected = trade_republic.parse(df)
    assert rejected == []
    assert transactions[0].side == "OTHER"
    assert transactions[0].raw_type == "SOME_NEW_TYPE_NOT_YET_SEEN"


def test_buy_with_non_numeric_shares_is_rejected_with_a_reason():
    df = pd.DataFrame([_row(shares="not-a-number", transaction_id="tx-bad")])
    transactions, rejected = trade_republic.parse(df)
    assert transactions == []
    assert len(rejected) == 1
    assert "shares" in rejected[0]["reason"]


def test_buy_with_no_symbol_or_name_is_rejected():
    df = pd.DataFrame([_row(symbol="", name="", transaction_id="tx-noid")])
    transactions, rejected = trade_republic.parse(df)
    assert transactions == []
    assert len(rejected) == 1


def test_missing_currency_is_rejected():
    df = pd.DataFrame([_row(currency="", transaction_id="tx-nocur")])
    _, rejected = trade_republic.parse(df)
    assert len(rejected) == 1


def test_duplicate_transaction_id_is_rejected_not_silently_merged():
    df = pd.DataFrame([_row(transaction_id="dup"), _row(transaction_id="dup", shares="2.0")])
    transactions, rejected = trade_republic.parse(df)
    assert len(transactions) == 1  # first one kept
    assert len(rejected) == 1  # second flagged, not silently dropped or merged


def test_multiple_currencies_in_one_export():
    df = pd.DataFrame([
        _row(currency="EUR", transaction_id="tx-eur"),
        _row(currency="USD", symbol="US6541061031", transaction_id="tx-usd"),
    ])
    transactions, rejected = trade_republic.parse(df)
    assert rejected == []
    assert {tx.currency for tx in transactions} == {"EUR", "USD"}


def test_missing_required_column_raises_not_a_trade_republic_export():
    df = pd.DataFrame([{"foo": "bar"}])
    with pytest.raises(ValueError):
        trade_republic.parse(df)


def test_one_bad_row_does_not_abort_the_whole_batch():
    df = pd.DataFrame([_row(transaction_id="tx-good"), _row(shares="bad", transaction_id="tx-bad")])
    transactions, rejected = trade_republic.parse(df)
    assert len(transactions) == 1
    assert len(rejected) == 1
