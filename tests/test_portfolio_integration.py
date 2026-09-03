"""Integration tests: broker CSV / simple CSV -> analyzable positions -> recommendations.

No network, no LLM, no Ollama. Price/sector lookups and research are faked at
the module boundary (`portfolio.price_provider`, `app.research_engine`) rather
than by hitting real providers; `signal_extraction.run_extraction` is replaced
by a call-counting fake wherever a test needs to prove no new LLM calls occur.
"""

from __future__ import annotations

import pandas as pd
import pytest

import app
import config
import portfolio as portfolio_mod


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

def _fake_price_history(ticker, provider=None):
    return pd.DataFrame({"adj_close": [100.0], "volume": [1_000_000]}, index=pd.to_datetime(["2026-01-01"]))


def _fake_research_with_signal(ticker):
    return {
        "ticker": ticker, "signal_count": 10, "research_score": 0.6,
        "risks": ["Elevated competitive pressure."], "catalysts": ["Strong revenue growth reported."],
        "signals_by_type": {}, "market_features": {}, "data_freshness": "2026-01-01",
    }


def _tr_row(**overrides):
    # Trade Republic's `symbol` is an ISIN for stocks/funds -- not ticker-shaped, so
    # NOT analyzable under the (deliberately preserved) no-security-master-resolution
    # rule. `BTC` is used here as the "analyzable" default because it's the one real
    # case where Trade Republic's own `symbol` happens to already be ticker-shaped
    # (crypto). Tests specifically exercising the ISIN/unmapped path use their own
    # ISIN symbol explicitly (see test_unmapped_fund_isin_is_visible_but_excluded...).
    base = {
        "datetime": "2024-10-09T09:15:36.798Z", "type": "BUY", "asset_class": "CRYPTO",
        "name": "Bitcoin", "symbol": "BTC", "shares": "2.0", "price": "150.0",
        "amount": "-300.0", "fee": "-1.0", "tax": "", "currency": "EUR", "transaction_id": "tx-1",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test in this file: no real price/sector lookups."""
    monkeypatch.setattr(portfolio_mod.price_provider, "get_price_history", _fake_price_history)
    monkeypatch.setattr(portfolio_mod, "_sector_for", lambda ticker: None)


@pytest.fixture
def no_extraction(monkeypatch, tmp_path):
    """Point config at tmp paths so _ensure_research_available never touches real data,
    and replace run_extraction with a call counter so a test can assert it was never called.
    """
    monkeypatch.setattr(config, "DOCUMENTS_PATH", tmp_path / "does-not-exist.parquet")
    monkeypatch.setattr(config, "SIGNALS_PATH", tmp_path / "signals.parquet")

    calls = {"count": 0}

    def _fail_if_called(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("signal_extraction.run_extraction should not be called in this test")

    monkeypatch.setattr(app.signal_extraction, "run_extraction", _fail_if_called)
    return calls


# ---------------------------------------------------------------------------
# 1. Simple portfolio CSV still works
# ---------------------------------------------------------------------------

def test_simple_csv_is_detected_and_analyzable():
    df = pd.DataFrame([{"ticker": "AAPL", "quantity": 10, "average_cost": 180.0}])
    result = app.import_and_prepare_portfolio(df)
    assert result["input_format"] == "canonical"
    assert result["analyzable_positions"]["ticker"].tolist() == ["AAPL"]
    assert result["unmapped_positions"].empty


# ---------------------------------------------------------------------------
# 2. Trade Republic CSV works
# ---------------------------------------------------------------------------

def test_trade_republic_csv_is_detected_and_normalized():
    df = pd.DataFrame([_tr_row()])
    result = app.import_and_prepare_portfolio(df)
    assert result["input_format"] == "trade_republic"
    assert result["transaction_count"] == 1
    assert result["n_trades"] == 1
    assert result["n_other"] == 0
    assert len(result["analyzable_positions"]) == 1
    row = result["analyzable_positions"].iloc[0]
    assert row["ticker"] == "BTC"
    assert row["quantity"] == 2.0
    assert row["average_cost"] == 150.0


# ---------------------------------------------------------------------------
# 3. Both produce compatible portfolio representations
# ---------------------------------------------------------------------------

def test_both_input_paths_produce_the_same_shape():
    simple = app.import_and_prepare_portfolio(
        pd.DataFrame([{"ticker": "AAPL", "quantity": 10, "average_cost": 180.0}])
    )
    tr = app.import_and_prepare_portfolio(pd.DataFrame([_tr_row()]))
    assert list(simple["analyzable_positions"].columns) == list(tr["analyzable_positions"].columns)
    assert set(simple["analyzable_positions"].columns) >= {"ticker", "quantity", "average_cost", "currency"}


# ---------------------------------------------------------------------------
# 4. Imported (normalized) portfolio reaches recommendation logic
# ---------------------------------------------------------------------------

def test_normalized_trade_republic_portfolio_reaches_recommendations(monkeypatch, no_extraction):
    monkeypatch.setattr(app.research_engine, "load_company_research", _fake_research_with_signal)

    import_result = app.import_and_prepare_portfolio(pd.DataFrame([_tr_row()]))
    result = app.analyze_portfolio(import_result["analyzable_positions"], "medium_term", "moderate")

    assert len(result["positions"]) == 1
    rec = result["positions"][0]["recommendation"]
    assert rec.ticker == "BTC"
    assert rec.action in ("BUY", "HOLD", "SELL", "WATCH", "INSUFFICIENT_EVIDENCE")
    assert no_extraction["count"] == 0  # no LLM call triggered by the importer or the analysis


# ---------------------------------------------------------------------------
# 5. Unsupported instruments remain visible but are not analyzed
# ---------------------------------------------------------------------------

def test_unmapped_fund_isin_is_visible_but_excluded_from_analysis():
    df = pd.DataFrame(
        [
            _tr_row(transaction_id="tx-stock"),  # a real, ticker-shaped position
            _tr_row(
                transaction_id="tx-fund", name="Core MSCI World USD (Acc)", asset_class="FUND",
                symbol="IE00B4L5Y983",  # ISIN, not a ticker -- no adapter maps funds to tickers
                shares="1.0", price="100.0", amount="-100.0",
            ),
        ]
    )
    result = app.import_and_prepare_portfolio(df)

    assert len(result["normalized_positions"]) == 2  # both reconstructed, rich (name/asset_class) view
    assert "IE00B4L5Y983" not in result["analyzable_positions"]["ticker"].tolist()

    unmapped = result["unmapped_positions"]
    assert "IE00B4L5Y983" in unmapped["instrument_id"].tolist()  # visible, not dropped
    unmapped_row = unmapped[unmapped["instrument_id"] == "IE00B4L5Y983"].iloc[0]
    assert unmapped_row["name"] == "Core MSCI World USD (Acc)"  # the point: not just a bare ISIN
    assert unmapped_row["asset_class"] == "FUND"
    assert any("Invalid ticker" in e for e in result["validation_errors"])


# ---------------------------------------------------------------------------
# 6. Empty/invalid portfolio produces a clear error
# ---------------------------------------------------------------------------

def test_empty_canonical_csv_produces_a_clear_message_not_a_crash():
    df = pd.DataFrame(columns=["ticker", "quantity", "average_cost"])
    result = app.import_and_prepare_portfolio(df)
    assert result["analyzable_positions"].empty
    assert result["validation_errors"] == ["No current positions were found in this file."]


def test_unrecognized_csv_format_produces_a_clear_message_not_a_crash():
    df = pd.DataFrame([{"foo": 1, "bar": 2}])
    result = app.import_and_prepare_portfolio(df)
    assert result["input_format"] == "unknown"
    assert result["validation_errors"]


def test_trade_republic_export_with_only_cash_rows_has_zero_positions():
    df = pd.DataFrame(
        [
            {
                "datetime": "2024-01-01T00:00:00Z", "type": "CUSTOMER_INBOUND", "asset_class": "",
                "name": "ANTONIN BRENGETTO", "symbol": "", "shares": "", "price": "",
                "amount": "100.0", "fee": "", "tax": "", "currency": "EUR", "transaction_id": "tx-cash",
            }
        ]
    )
    result = app.import_and_prepare_portfolio(df)
    assert result["input_format"] == "trade_republic"
    assert result["n_trades"] == 0
    assert result["analyzable_positions"].empty
    assert "No current positions" in result["validation_errors"][0]


# ---------------------------------------------------------------------------
# 7. Re-uploading the same portfolio does not trigger new LLM calls when cached
# ---------------------------------------------------------------------------

def test_reupload_with_cached_research_triggers_zero_llm_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DOCUMENTS_PATH", tmp_path / "documents.parquet")
    config.DOCUMENTS_PATH.touch()  # just needs to exist
    signals_path = tmp_path / "signals.parquet"
    pd.DataFrame([{"ticker": "BTC", "signal_type": "risk", "direction": "negative", "strength": 0.5}]).to_parquet(signals_path)
    monkeypatch.setattr(config, "SIGNALS_PATH", signals_path)

    calls = {"count": 0}
    monkeypatch.setattr(app.signal_extraction, "run_extraction", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1))

    import_result = app.import_and_prepare_portfolio(pd.DataFrame([_tr_row()]))

    app._ensure_research_available(import_result["analyzable_positions"]["ticker"].tolist())
    app._ensure_research_available(import_result["analyzable_positions"]["ticker"].tolist())  # simulate re-upload

    assert calls["count"] == 0


# ---------------------------------------------------------------------------
# 8. Portfolio-level calculations remain deterministic
# ---------------------------------------------------------------------------

def test_import_is_deterministic_across_repeated_runs():
    df = pd.DataFrame([_tr_row(), _tr_row(transaction_id="tx-2", shares="1.0", price="200.0", amount="-200.0")])
    first = app.import_and_prepare_portfolio(df)
    second = app.import_and_prepare_portfolio(df)
    pd.testing.assert_frame_equal(
        first["analyzable_positions"].reset_index(drop=True),
        second["analyzable_positions"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 9. Recommendation output retains the four separate concepts end-to-end
# ---------------------------------------------------------------------------

def test_recommendation_keeps_four_concepts_separate_through_full_flow(monkeypatch, no_extraction):
    monkeypatch.setattr(app.research_engine, "load_company_research", _fake_research_with_signal)

    import_result = app.import_and_prepare_portfolio(pd.DataFrame([_tr_row()]))
    result = app.analyze_portfolio(import_result["analyzable_positions"], "medium_term", "moderate")

    rec = result["positions"][0]["recommendation"]
    assert rec.asset_signal is not None
    assert rec.strategy_fit is not None or rec.strategy_fit is None  # present as its own field either way
    assert hasattr(rec, "portfolio_fit")
    assert isinstance(rec.key_risks, list)
    assert rec.data_freshness is not None
    # not collapsed into one field -- each concept independently addressable
    assert {"asset_signal", "strategy_fit", "portfolio_fit"}.issubset(rec.model_dump().keys())
