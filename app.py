"""Streamlit MVP: upload a portfolio, run analysis, see recommendations.

This file only orchestrates -- every real computation (validation, position
math, cache-first signal extraction, research aggregation, strategy scoring,
recommendation generation) lives in `src/`. `analyze_portfolio()` takes the
portfolio as a plain argument (never a module-level global), so the exact same
function serves any number of portfolios.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

import config
import portfolio as portfolio_mod
import recommendations
import research_engine
import signal_extraction
import strategy
from portfolio_importers import detect, trade_republic
from portfolio_importers.schema import reconstruct_positions, to_simple_portfolio

MAX_AUTO_CHUNKS_PER_MISSING_TICKER = 10  # safety cap for on-demand extraction from the UI

# Broker adapters keyed by what detect.detect_format() returns. Adding a new
# broker: write its adapter module (see README "how to add a new broker
# adapter"), add a branch to detect.detect_format(), and add it here.
IMPORTERS = {"trade_republic": trade_republic}


def _ensure_research_available(tickers: list[str]) -> dict:
    """Run cache-first extraction only for tickers with no signals at all yet.

    This is step 6 of the workflow: a ticker 100 users already hold triggers
    zero new LLM calls here, because `signal_extraction` is cache-first and
    this function only ever calls it for tickers with nothing cached.

    Each missing ticker gets its own chunk budget, run separately -- a single
    combined call across several missing tickers would let one ticker (sorted
    first alphabetically) exhaust the whole budget and starve the rest.
    Tickers with no SEC documents at all (e.g. foreign filers not fetched via
    fetch_edgar.py) simply produce zero chunks and are skipped, not an error.
    """
    if not config.DOCUMENTS_PATH.exists():
        return {"ran_extraction_for": [], "note": "No processed SEC documents available."}

    existing_signals = pd.read_parquet(config.SIGNALS_PATH) if config.SIGNALS_PATH.exists() else pd.DataFrame()
    covered = set(existing_signals["ticker"].unique()) if not existing_signals.empty else set()
    missing = [t for t in tickers if t.upper() not in covered]

    if not missing:
        return {"ran_extraction_for": []}

    docs = pd.read_parquet(config.DOCUMENTS_PATH)
    run_summaries = {}
    for ticker in missing:
        new_signals, run_summary = signal_extraction.run_extraction(
            docs, source="sec", tickers=[ticker], max_chunks=MAX_AUTO_CHUNKS_PER_MISSING_TICKER,
        )
        signal_extraction.save_signals(new_signals, config.SIGNALS_PATH)
        run_summaries[ticker] = run_summary

    return {"ran_extraction_for": missing, "run_summary": run_summaries}


def import_and_prepare_portfolio(raw_df: pd.DataFrame) -> dict:
    """Detect format, normalize a broker export if needed, and validate.

    Pure Python, no Streamlit, no LLM, no network -- this is the one place a
    simple portfolio CSV and a supported broker export converge into the same
    `ticker, quantity, average_cost, currency` shape `analyze_portfolio()`
    already consumes unchanged. It does not call `analyze_portfolio()` itself,
    so normalization stays fully separate from recommendation logic; the
    caller decides whether/when to run analysis on the result.

    `analyzable_positions` is `portfolio.validate_portfolio()`'s clean output --
    this is what should be passed to `analyze_portfolio()`. `normalized_positions`
    and `unmapped_positions` keep the richer canonical fields (name, asset_class,
    instrument_id) where available, specifically so a user can tell *what* an
    unmapped instrument actually is (e.g. "Core MSCI World USD (Acc), FUND"),
    not just see a bare unrecognized ticker/ISIN string -- a bare string is not
    enough for the transparency this exists for. Nothing normalized is ever
    silently dropped; every problem row has its reason in `validation_errors`.
    """
    empty_report = {
        "input_format": None, "transaction_count": None, "n_trades": None, "n_other": None,
        "rejected_rows": [], "normalized_positions": pd.DataFrame(),
        "analyzable_positions": pd.DataFrame(), "unmapped_positions": pd.DataFrame(),
        "validation_errors": [],
    }

    input_format = detect.detect_format(raw_df)
    report = {**empty_report, "input_format": input_format}

    if input_format == "unknown":
        report["validation_errors"] = [
            "Unrecognized CSV format -- not a supported broker export or the plain portfolio CSV schema."
        ]
        return report

    if input_format == "trade_republic":
        adapter = IMPORTERS["trade_republic"]
        canonical_transactions, rejected_rows = adapter.parse(raw_df)
        report["transaction_count"] = len(raw_df)
        report["n_trades"] = sum(1 for t in canonical_transactions if t.side in ("BUY", "SELL"))
        report["n_other"] = len(canonical_transactions) - report["n_trades"]
        report["rejected_rows"] = rejected_rows

        positions = reconstruct_positions(canonical_transactions)
        # Rich, display-oriented view (kept in the same order as `simple`, so the
        # two align positionally for the analyzable/unmapped split below).
        display_df = pd.DataFrame([p.model_dump() for p in positions])
        simple = to_simple_portfolio(positions)  # bridged ticker/quantity/average_cost/currency
    else:  # canonical -- already portfolio.py's own schema, nothing richer to show
        display_df = raw_df.copy()
        simple = raw_df.copy()
        if "currency" not in simple.columns:
            simple["currency"] = "USD"
        if "currency" not in display_df.columns:
            display_df["currency"] = "USD"

    report["normalized_positions"] = display_df

    if simple.empty:
        report["analyzable_positions"] = simple
        report["validation_errors"] = ["No current positions were found in this file."]
        return report

    clean, errors = portfolio_mod.validate_portfolio(simple)
    is_analyzable = simple["ticker"].astype(str).str.strip().str.upper().isin(clean["ticker"]).to_numpy()

    report["analyzable_positions"] = clean
    report["unmapped_positions"] = display_df[~is_analyzable]  # rich view, for display only
    report["validation_errors"] = errors
    return report


def analyze_portfolio(portfolio_df: pd.DataFrame, chosen_strategy: str, risk_profile: str) -> dict:
    """Pure orchestration: validate -> enrich -> research -> strategy -> recommend."""
    clean, errors = portfolio_mod.validate_portfolio(portfolio_df)

    extraction_note = _ensure_research_available(clean["ticker"].tolist()) if not clean.empty else {}

    enriched = portfolio_mod.enrich_positions(clean) if not clean.empty else clean
    concentration = portfolio_mod.portfolio_concentration(enriched) if not enriched.empty else {}
    sectors = portfolio_mod.sector_exposure(enriched) if not enriched.empty else {"available": False, "exposure": {}}

    holdings_weights = dict(zip(enriched.get("ticker", []), enriched.get("portfolio_weight", [])))
    portfolio_context = {"holdings": holdings_weights}

    position_results = []
    for _, row in enriched.iterrows():
        research = research_engine.load_company_research(row["ticker"])
        fit = strategy.score_strategy_fit(research, chosen_strategy)
        rec = recommendations.generate_recommendation(
            row["ticker"], research, fit, portfolio_context, risk_profile, chosen_strategy,
        )
        position_results.append({"position": row.to_dict(), "research": research, "strategy_fit": fit, "recommendation": rec})

    return {
        "errors": errors,
        "enriched": enriched,
        "concentration": concentration,
        "sectors": sectors,
        "positions": position_results,
        "extraction_note": extraction_note,
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Quant Research Lab", layout="wide")
st.title("Quant Research Lab -- Portfolio Analysis")
st.caption(
    "Research signal, not personalized financial advice. Every recommendation below is a model "
    "output with stated evidence, confidence, and data freshness -- not a promise of performance."
)

st.header("1. Import your portfolio")
st.caption(
    "Upload either a broker transaction export or a plain portfolio CSV (ticker, quantity, "
    "average_cost[, currency]) -- the format is detected automatically from its columns, not "
    "its filename."
)
uploaded = st.file_uploader(
    "Upload CSV", type="csv", key="portfolio_upload",
    help="Only CSV is supported today -- a PDF broker statement will be rejected with a clear "
    "message, not parsed.",
)
risk_profile = st.selectbox("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
chosen_strategy = st.selectbox("Strategy", ["short_term", "medium_term", "long_term"], index=1)
run = st.button("Run analysis", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(
            f"Couldn't read this file as CSV ({type(exc).__name__}). If this is a PDF broker "
            "statement, PDF parsing isn't supported yet -- please export a CSV from your broker "
            "instead."
        )
        st.stop()

    import_result = import_and_prepare_portfolio(raw_df)

    st.header("2. Import summary")
    st.write(f"**Detected format:** `{import_result['input_format']}`")

    if import_result["input_format"] == "unknown":
        st.error(import_result["validation_errors"][0])
        st.stop()

    metric_cols = st.columns(4)
    if import_result["transaction_count"] is not None:
        metric_cols[0].metric("Transaction rows", import_result["transaction_count"])
        metric_cols[1].metric("Trade transactions", import_result["n_trades"])
    else:
        metric_cols[0].metric("Input rows", len(raw_df))
    metric_cols[2].metric("Current positions", len(import_result["normalized_positions"]))
    metric_cols[3].metric("Analyzable positions", len(import_result["analyzable_positions"]))

    if import_result["rejected_rows"]:
        with st.expander(f"{len(import_result['rejected_rows'])} rejected transaction row(s) -- reasons"):
            st.dataframe(pd.DataFrame(import_result["rejected_rows"]))

    if import_result["validation_errors"]:
        st.warning(
            "Import/normalization warnings:\n\n"
            + "\n".join(f"- {e}" for e in import_result["validation_errors"])
        )

    st.write("**Normalized positions:**")
    st.dataframe(import_result["normalized_positions"])

    if not import_result["unmapped_positions"].empty:
        st.warning(
            f"{len(import_result['unmapped_positions'])} position(s) are **not analyzable** "
            "(unmapped/unsupported instrument or invalid data -- see warnings above) and are "
            "excluded from recommendations below. They are not silently dropped: this is the "
            "full list of what was found but couldn't be analyzed."
        )
        st.dataframe(import_result["unmapped_positions"])

    if import_result["analyzable_positions"].empty:
        st.error("No analyzable positions -- nothing to run recommendations on.")
        st.stop()

    st.divider()
    st.header("3. Portfolio analysis")

    with st.spinner("Analyzing portfolio (only uncached tickers trigger new research)..."):
        result = analyze_portfolio(import_result["analyzable_positions"], chosen_strategy, risk_profile)

    enriched = result["enriched"]
    if enriched.empty:
        st.error("No valid positions to analyze.")
    else:
        st.subheader("Portfolio overview")
        total_value = enriched["market_value"].sum(skipna=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total value", f"${total_value:,.0f}" if pd.notna(total_value) else "n/a")
        col2.metric("Positions", len(enriched))
        top_weight = result["concentration"].get("top_n_weight")
        col3.metric("Top 3 concentration", f"{top_weight:.0%}" if top_weight is not None else "n/a")

        missing_price = enriched[enriched["current_price"].isna()]
        if not missing_price.empty:
            st.info(
                f"**Missing market data** for {len(missing_price)} analyzable position(s) -- "
                f"{sorted(missing_price['ticker'].tolist())}. These have a valid ticker shape but "
                "no price could be fetched (e.g. delisted, wrong exchange suffix, or a temporary "
                "data-provider gap). They still appear below with recommendations based on "
                "research signals, just without P&L/weight numbers."
            )

        if result["sectors"]["available"]:
            st.write("**Sector exposure**")
            st.bar_chart(pd.Series(result["sectors"]["exposure"]))
        else:
            st.caption("Sector exposure: unavailable for this data source.")

        st.subheader("Holdings")
        st.dataframe(
            enriched[["ticker", "quantity", "current_price", "market_value", "portfolio_weight", "unrealized_pl_pct", "sector"]]
        )

        st.subheader("Recommendations")
        for item in result["positions"]:
            rec = item["recommendation"]
            label = f"{rec.ticker} -- {rec.action} (confidence {rec.confidence:.0%})"
            if rec.action == "INSUFFICIENT_EVIDENCE":
                label = f"{rec.ticker} -- ⚠️ insufficient evidence for a recommendation"
            with st.expander(label):
                c1, c2, c3 = st.columns(3)
                c1.metric("Asset signal", f"{rec.asset_signal:.2f}" if rec.asset_signal is not None else "n/a")
                c2.metric("Strategy fit", f"{rec.strategy_fit:.2f}" if rec.strategy_fit is not None else "n/a")
                c3.metric("Portfolio fit", f"{rec.portfolio_fit:.2f}" if rec.portfolio_fit is not None else "n/a")
                st.write(f"**Bull case:** {rec.bull_case}")
                st.write(f"**Bear case:** {rec.bear_case}")
                st.write(f"**Portfolio consideration:** {rec.portfolio_consideration}")
                st.write("**Key signals:** " + "; ".join(rec.key_signals))
                st.write("**Key risks:** " + "; ".join(rec.key_risks))
                st.caption(f"Data freshness: {rec.data_freshness or 'unknown'}")

        extraction_note = result.get("extraction_note", {})
        if extraction_note.get("ran_extraction_for"):
            st.caption(f"Ran new research for previously-uncached tickers: {extraction_note['ran_extraction_for']}")
        else:
            st.caption("All analyzed tickers were already cached -- no new research/LLM calls were made.")
