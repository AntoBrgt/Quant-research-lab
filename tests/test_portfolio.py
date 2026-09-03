"""Portfolio CSV validation and position math -- no network required.

`validate_portfolio` is pure pandas and is tested directly with no network.
`enrich_positions` is tested with a stub PriceProvider so it never calls
yfinance either.
"""

import pandas as pd
import pytest

import portfolio


def test_valid_portfolio_passes_with_no_errors():
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "quantity": 10, "average_cost": 180, "currency": "USD"},
            {"ticker": "MSFT", "quantity": 5, "average_cost": 400, "currency": "USD"},
        ]
    )
    clean, errors = portfolio.validate_portfolio(df)
    assert errors == []
    assert len(clean) == 2


def test_missing_ticker_is_dropped_and_reported():
    df = pd.DataFrame([{"ticker": "", "quantity": 10, "average_cost": 180, "currency": "USD"}])
    clean, errors = portfolio.validate_portfolio(df)
    assert clean.empty
    assert any("missing ticker" in e for e in errors)


def test_invalid_ticker_is_dropped_and_reported():
    df = pd.DataFrame([{"ticker": "NOT A TICKER!!", "quantity": 10, "average_cost": 180, "currency": "USD"}])
    clean, errors = portfolio.validate_portfolio(df)
    assert clean.empty
    assert any("Invalid ticker" in e for e in errors)


def test_negative_quantity_is_dropped_and_reported():
    df = pd.DataFrame([{"ticker": "AAPL", "quantity": -5, "average_cost": 180, "currency": "USD"}])
    clean, errors = portfolio.validate_portfolio(df)
    assert clean.empty
    assert any("Negative quantity" in e for e in errors)


def test_duplicate_ticker_is_dropped_and_reported():
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "quantity": 10, "average_cost": 180, "currency": "USD"},
            {"ticker": "AAPL", "quantity": 5, "average_cost": 190, "currency": "USD"},
        ]
    )
    clean, errors = portfolio.validate_portfolio(df)
    assert clean.empty
    assert any("Duplicate ticker" in e for e in errors)


def test_missing_average_cost_is_dropped_and_reported():
    df = pd.DataFrame([{"ticker": "AAPL", "quantity": 10, "average_cost": None, "currency": "USD"}])
    clean, errors = portfolio.validate_portfolio(df)
    assert clean.empty
    assert any("average_cost" in e for e in errors)


def test_mixed_valid_and_invalid_rows_keeps_only_valid():
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "quantity": 10, "average_cost": 180, "currency": "USD"},
            {"ticker": "BAD TICKER", "quantity": 5, "average_cost": 100, "currency": "USD"},
        ]
    )
    clean, errors = portfolio.validate_portfolio(df)
    assert clean["ticker"].tolist() == ["AAPL"]
    assert len(errors) == 1


class StubPriceProvider:
    """Fixed prices, no network -- for enrich_positions tests."""

    def __init__(self, prices: dict[str, float]):
        self._prices = prices

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        price = self._prices[ticker.upper()]
        return pd.DataFrame({"adj_close": [price], "volume": [1_000_000]}, index=pd.to_datetime(["2026-01-01"]))


def test_enrich_positions_computes_weight_and_pl():
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "quantity": 10, "average_cost": 100, "currency": "USD"},
            {"ticker": "MSFT", "quantity": 10, "average_cost": 100, "currency": "USD"},
        ]
    )
    stub = StubPriceProvider({"AAPL": 200, "MSFT": 100})
    enriched = portfolio.enrich_positions(df, price_prov=stub, include_sector=False)

    aapl = enriched[enriched["ticker"] == "AAPL"].iloc[0]
    assert aapl["market_value"] == 2000
    assert aapl["unrealized_pl"] == 1000
    assert aapl["unrealized_pl_pct"] == pytest.approx(1.0)
    # AAPL 2000 / total 3000 = 2/3 weight
    assert aapl["portfolio_weight"] == pytest.approx(2 / 3)


def test_concentration_is_higher_for_single_position():
    df = pd.DataFrame([{"ticker": "AAPL", "quantity": 10, "average_cost": 100, "currency": "USD"}])
    stub = StubPriceProvider({"AAPL": 100})
    enriched = portfolio.enrich_positions(df, price_prov=stub, include_sector=False)
    concentration = portfolio.portfolio_concentration(enriched)
    assert concentration["hhi"] == pytest.approx(1.0)  # single position = fully concentrated


def test_sector_exposure_reports_unavailable_explicitly():
    df = pd.DataFrame([{"ticker": "AAPL", "quantity": 10, "average_cost": 100, "currency": "USD"}])
    stub = StubPriceProvider({"AAPL": 100})
    enriched = portfolio.enrich_positions(df, price_prov=stub, include_sector=False)
    sectors = portfolio.sector_exposure(enriched)
    assert sectors["available"] is False
