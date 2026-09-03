"""Column/schema-based format detection -- no filenames, no network."""

import pandas as pd

from portfolio_importers import detect


def test_trade_republic_export_is_detected():
    df = pd.DataFrame(columns=[
        "datetime", "date", "account_type", "category", "type", "asset_class",
        "name", "symbol", "shares", "price", "amount", "fee", "tax", "currency",
    ])
    assert detect.detect_format(df) == "trade_republic"


def test_canonical_portfolio_csv_is_detected():
    df = pd.DataFrame(columns=["ticker", "quantity", "average_cost", "currency"])
    assert detect.detect_format(df) == "canonical"


def test_canonical_portfolio_csv_without_optional_currency_is_still_detected():
    df = pd.DataFrame(columns=["ticker", "quantity", "average_cost"])
    assert detect.detect_format(df) == "canonical"


def test_unrelated_csv_is_unknown():
    df = pd.DataFrame(columns=["foo", "bar", "baz"])
    assert detect.detect_format(df) == "unknown"


def test_empty_csv_is_unknown():
    df = pd.DataFrame()
    assert detect.detect_format(df) == "unknown"


def test_partial_trade_republic_columns_do_not_false_positive():
    # Has some overlapping-sounding columns but not the full required set.
    df = pd.DataFrame(columns=["datetime", "type", "amount"])
    assert detect.detect_format(df) == "unknown"
