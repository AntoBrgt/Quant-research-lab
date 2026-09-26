"""Fundamental engine: historical series, provenance, quality flags, trend
classification, and the flat output schema -- all deterministic Python over a
fake provider, no network dependency anywhere in this file (section 13).
"""

import fundamentals


class FakeFundamentalsProvider:
    def __init__(self, raw: dict):
        self._raw = raw

    def get_raw_fundamentals(self, ticker: str) -> dict:
        return self._raw


def _raw(info=None, income_stmt=None, cashflow=None, balance_sheet=None):
    return {
        "info": info or {}, "income_stmt": income_stmt or {},
        "cashflow": cashflow or {}, "balance_sheet": balance_sheet or {},
    }


# ---------------------------------------------------------------------------
# Missing data (section 7): never fabricated, never zero-filled
# ---------------------------------------------------------------------------

def test_missing_fields_are_none_not_fabricated():
    provider = FakeFundamentalsProvider(_raw())
    result = fundamentals.compute_fundamentals("NOPE", provider=provider)

    assert result["revenue_growth"] is None
    assert result["roic"] is None
    assert result["interest_coverage"] is None
    assert result["pe"] is None
    assert result["data_quality"] == []


def test_missing_metrics_are_none_not_zero():
    provider = FakeFundamentalsProvider(_raw(info={"sector": "Technology"}))
    result = fundamentals.compute_fundamentals("X", provider=provider)
    # A metric with genuinely no data must be None, never silently 0.0.
    assert result["revenue_growth"] is None
    assert result["free_cash_flow"] is None


# ---------------------------------------------------------------------------
# Metric normalization / valuation (direct info passthrough)
# ---------------------------------------------------------------------------

def test_direct_info_fields_pass_through():
    provider = FakeFundamentalsProvider(
        _raw(info={
            "sector": "Technology", "industry": "Consumer Electronics", "marketCap": 3_000_000_000_000,
            "totalDebt": 100_000_000_000, "trailingPE": 30.0, "forwardPE": 27.0,
            "priceToBook": 45.0, "enterpriseToEbitda": 22.0, "priceToSalesTrailing12Months": 9.5,
        })
    )
    result = fundamentals.compute_fundamentals("AAPL", provider=provider)

    assert result["sector"] == "Technology"
    assert result["market_cap"] == 3_000_000_000_000
    assert result["pe"] == 30.0
    assert result["forward_pe"] == 27.0
    assert result["price_book"] == 45.0
    assert result["ev_ebitda"] == 22.0
    assert result["price_sales"] == 9.5


# ---------------------------------------------------------------------------
# Historical periods + growth calculations
# ---------------------------------------------------------------------------

def test_revenue_growth_series_covers_every_period_not_just_latest():
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={"Total Revenue": {"2023-12-31": 100.0, "2024-12-31": 118.0, "2025-12-31": 146.0}})
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)

    series = result["history"]["revenue_growth"]
    assert [o["period"] for o in series] == ["FY2024", "FY2025"]
    assert round(series[0]["value"], 4) == 0.18
    assert round(series[1]["value"], 4) == round((146.0 - 118.0) / 118.0, 4)
    # The flat field reflects the latest period, not an average or the first.
    assert round(result["revenue_growth"], 4) == round((146.0 - 118.0) / 118.0, 4)


def test_gross_and_net_margin_calculations():
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={
            "Total Revenue": {"2025-12-31": 200.0},
            "Gross Profit": {"2025-12-31": 90.0},
            "Net Income": {"2025-12-31": 40.0},
        })
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["gross_margin"] == 0.45
    assert result["net_margin"] == 0.20


# ---------------------------------------------------------------------------
# FCF calculations
# ---------------------------------------------------------------------------

def test_fcf_derived_from_operating_cashflow_minus_capex_when_no_direct_row():
    provider = FakeFundamentalsProvider(
        _raw(cashflow={
            "Operating Cash Flow": {"2025-12-31": 100.0, "2024-12-31": 90.0},
            "Capital Expenditure": {"2025-12-31": -20.0, "2024-12-31": -20.0},
        })
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["free_cash_flow"] == 80.0  # 100 - 20
    assert round(result["fcf_growth"], 4) == round((80.0 - 70.0) / 70.0, 4)


def test_direct_free_cash_flow_row_preferred_over_derived():
    provider = FakeFundamentalsProvider(
        _raw(cashflow={
            "Free Cash Flow": {"2025-12-31": 999.0},
            "Operating Cash Flow": {"2025-12-31": 100.0},
            "Capital Expenditure": {"2025-12-31": -20.0},
        })
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["free_cash_flow"] == 999.0


def test_fcf_yield_uses_market_cap():
    provider = FakeFundamentalsProvider(
        _raw(info={"marketCap": 1000.0}, cashflow={"Free Cash Flow": {"2025-12-31": 50.0}})
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["fcf_yield"] == 0.05


# ---------------------------------------------------------------------------
# Duplicate periods / impossible values (section 8) -- flagged, not dropped
# ---------------------------------------------------------------------------

def test_impossible_margin_is_flagged_not_dropped():
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={
            "Total Revenue": {"2025-12-31": 10.0},
            "Net Income": {"2025-12-31": 25.0},  # net income > revenue -> 250% margin
        })
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["net_margin"] == 2.5  # value is kept, not deleted or clamped
    assert any("impossible_margin" in f for f in result["data_quality"])


def test_wide_but_real_roe_is_not_flagged_as_impossible():
    """A >150% ROE is routine for a low-equity, buyback-heavy company (e.g.
    Apple) -- it must not be flagged the same way an impossible margin would.
    """
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={"Net Income": {"2025-12-31": 20.0}}, balance_sheet={"Stockholders Equity": {"2025-12-31": 10.0}})
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["roe"] == 2.0
    assert not any("roe" in f and "impossible" in f for f in result["data_quality"])


def test_implausible_ratio_is_still_flagged_when_truly_extreme():
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={"Net Income": {"2025-12-31": 500.0}}, balance_sheet={"Stockholders Equity": {"2025-12-31": 1.0}})
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["roe"] == 500.0
    assert any("implausible_ratio" in f for f in result["data_quality"])


def test_extreme_growth_value_is_flagged():
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={"Total Revenue": {"2024-12-31": 10.0, "2025-12-31": 100.0}})  # +900%
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["revenue_growth"] == 9.0
    assert any("extreme_growth_value" in f for f in result["data_quality"])


def test_duplicate_period_is_flagged_and_both_kept():
    observations = [
        fundamentals.MetricObservation(metric="x", value=1.0, period="FY2025", period_type="annual", period_end="2025-12-31", source="s", retrieved_at="t"),
        fundamentals.MetricObservation(metric="x", value=2.0, period="FY2025", period_type="annual", period_end="2025-12-31", source="s", retrieved_at="t"),
    ]
    fundamentals.validate_observations(observations, kind="raw")
    assert all("duplicate_period" in o.quality_flags for o in observations)
    assert len(observations) == 2  # neither is dropped


def test_future_dated_period_is_flagged():
    observations = [
        fundamentals.MetricObservation(metric="x", value=1.0, period="FY2099", period_type="annual", period_end="2099-01-01", source="s", retrieved_at="t"),
    ]
    fundamentals.validate_observations(observations, kind="raw")
    assert "future_dated_period" in observations[0].quality_flags


def test_invalid_date_is_flagged_not_raised():
    observations = [
        fundamentals.MetricObservation(metric="x", value=1.0, period="bad", period_type="annual", period_end="not-a-date", source="s", retrieved_at="t"),
    ]
    fundamentals.validate_observations(observations, kind="raw")
    assert "invalid_date" in observations[0].quality_flags


# ---------------------------------------------------------------------------
# Trend classification (section 9) -- descriptive states only
# ---------------------------------------------------------------------------

def _obs(*values):
    return [
        fundamentals.MetricObservation(metric="x", value=v, period=f"FY{2020+i}", period_type="annual", period_end=f"{2020+i}-12-31", source="s", retrieved_at="t")
        for i, v in enumerate(values)
    ]


def test_trend_insufficient_data_with_fewer_than_two_points():
    assert fundamentals.classify_trend(_obs(0.1)) == "INSUFFICIENT_DATA"
    assert fundamentals.classify_trend([]) == "INSUFFICIENT_DATA"


def test_trend_improving_with_consistent_rise():
    assert fundamentals.classify_trend(_obs(0.10, 0.15, 0.20, 0.24)) == "IMPROVING"


def test_trend_deteriorating_with_consistent_fall():
    assert fundamentals.classify_trend(_obs(0.30, 0.24, 0.18, 0.12)) == "DETERIORATING"


def test_trend_stable_with_flat_series():
    assert fundamentals.classify_trend(_obs(0.20, 0.205, 0.198, 0.202)) == "STABLE"


def test_trend_respects_higher_is_better_false_for_debt_like_metrics():
    # Debt rising every period is a DETERIORATING balance sheet even though
    # the raw values are increasing.
    assert fundamentals.classify_trend(_obs(10, 20, 30, 40), higher_is_better=False) == "DETERIORATING"
    assert fundamentals.classify_trend(_obs(40, 30, 20, 10), higher_is_better=False) == "IMPROVING"


def test_revenue_growth_trend_reflects_the_real_series():
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={"Total Revenue": {
            "2022-12-31": 100.0, "2023-12-31": 118.0, "2024-12-31": 143.0, "2025-12-31": 175.0,
        }})
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["revenue_growth_trend"] == "IMPROVING"


def test_balance_sheet_trend_uses_cash_not_a_fabricated_debt_history():
    """total_debt only has a current snapshot (see module docstring) -- the
    trend must come from cash (genuinely historical), not from combining a
    single current debt value with historical periods.
    """
    provider = FakeFundamentalsProvider(
        _raw(
            info={"totalDebt": 50.0},
            balance_sheet={"Cash And Cash Equivalents": {"2023-12-31": 10.0, "2024-12-31": 20.0, "2025-12-31": 35.0}},
        )
    )
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert result["balance_sheet_trend"] == "IMPROVING"
    assert len(result["history"]["net_debt"]) == 1  # current snapshot only, not a fake 3-period series


# ---------------------------------------------------------------------------
# Data freshness / provenance
# ---------------------------------------------------------------------------

def test_data_freshness_is_none_when_never_cached(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "FUNDAMENTALS_DIR", tmp_path / "fundamentals")
    provider = FakeFundamentalsProvider(_raw(info={"sector": "Technology"}))
    result = fundamentals.compute_fundamentals("NEVERCACHED", provider=provider)
    assert result["data_freshness"] is None


def test_every_observation_carries_source_and_period_type():
    provider = FakeFundamentalsProvider(_raw(income_stmt={"Total Revenue": {"2025-12-31": 100.0}}))
    result = fundamentals.compute_fundamentals("X", provider=provider)
    obs = result["history"]["revenue"][0]
    assert obs["source"] == "yfinance:financials"
    assert obs["period_type"] == "annual"
    assert obs["is_estimate"] is False
    assert obs["retrieved_at"]


def test_sources_list_only_includes_sources_actually_used():
    provider = FakeFundamentalsProvider(_raw(income_stmt={"Total Revenue": {"2025-12-31": 100.0}}))
    result = fundamentals.compute_fundamentals("X", provider=provider)
    assert "yfinance:financials" in result["sources"]
    assert "yfinance:cashflow" not in result["sources"]  # no cashflow data was actually provided


# ---------------------------------------------------------------------------
# Provider failure -- never raises, falls back honestly
# ---------------------------------------------------------------------------

class FailingProvider:
    def get_raw_fundamentals(self, ticker: str) -> dict:
        raise ConnectionError("network unavailable")


def test_provider_exception_does_not_propagate_from_compute_fundamentals():
    # A ticker whose provider call raises must not crash the whole research
    # pipeline -- same "one bad ticker doesn't take down a batch" principle
    # used throughout this project (research_engine's price fetch, portfolio's
    # sector lookup, signal_extraction's LLM call).
    result = fundamentals.compute_fundamentals("X", provider=FailingProvider())
    assert result["revenue_growth"] is None
    assert result["data_quality"] == []


def test_yfinance_provider_falls_back_to_stale_cache_on_fetch_failure(tmp_path, monkeypatch):
    import json as _json
    import os
    import time

    import config

    monkeypatch.setattr(config, "FUNDAMENTALS_DIR", tmp_path / "fundamentals")
    cache_path = tmp_path / "fundamentals" / "X.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(_json.dumps({"info": {"sector": "Technology"}, "income_stmt": {}, "cashflow": {}, "balance_sheet": {}}))
    old_time = time.time() - 100 * 3600  # force staleness so a real fetch is attempted
    os.utime(cache_path, (old_time, old_time))

    import yfinance

    class _RaisingTicker:
        def __init__(self, *_args, **_kwargs):
            raise ConnectionError("simulated network failure -- no real network call is made")

    monkeypatch.setattr(yfinance, "Ticker", _RaisingTicker)

    provider = fundamentals.YFinanceFundamentalsProvider()
    raw = provider.get_raw_fundamentals("X")
    assert raw["info"]["sector"] == "Technology"  # stale cache used, not an empty/raised result


# ---------------------------------------------------------------------------
# No metric forced onto every company (section 3/5)
# ---------------------------------------------------------------------------

def test_a_bank_with_no_gross_profit_line_just_gets_none_not_an_error():
    provider = FakeFundamentalsProvider(
        _raw(income_stmt={"Total Revenue": {"2025-12-31": 100.0}, "Net Income": {"2025-12-31": 20.0}})
    )
    result = fundamentals.compute_fundamentals("BANK", provider=provider)
    assert result["gross_margin"] is None  # no Gross Profit line at all -- not forced/guessed
    assert result["net_margin"] == 0.20  # still computed where the data exists
