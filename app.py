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

MAX_AUTO_CHUNKS_PER_MISSING_TICKER = 10  # safety cap for on-demand extraction from the UI


def _ensure_research_available(tickers: list[str]) -> dict:
    """Run cache-first extraction only for tickers with no signals at all yet.

    This is step 6 of the workflow: a ticker 100 users already hold triggers
    zero new LLM calls here, because `signal_extraction` is cache-first and
    this function only ever calls it for tickers with nothing cached.
    """
    if not config.DOCUMENTS_PATH.exists():
        return {"ran_extraction_for": [], "note": "No processed SEC documents available."}

    existing_signals = pd.read_parquet(config.SIGNALS_PATH) if config.SIGNALS_PATH.exists() else pd.DataFrame()
    covered = set(existing_signals["ticker"].unique()) if not existing_signals.empty else set()
    missing = [t for t in tickers if t.upper() not in covered]

    if not missing:
        return {"ran_extraction_for": []}

    docs = pd.read_parquet(config.DOCUMENTS_PATH)
    new_signals, run_summary = signal_extraction.run_extraction(
        docs, source="sec", tickers=missing, max_chunks=MAX_AUTO_CHUNKS_PER_MISSING_TICKER * len(missing),
    )

    if not new_signals.empty:
        combined = pd.concat([existing_signals, new_signals], ignore_index=True) if not existing_signals.empty else new_signals
        config.SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(config.SIGNALS_PATH, index=False)

    return {"ran_extraction_for": missing, "run_summary": run_summary}


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

uploaded = st.file_uploader("Upload portfolio CSV (ticker, quantity, average_cost[, currency])", type="csv")
risk_profile = st.selectbox("Risk profile", ["conservative", "moderate", "aggressive"], index=1)
chosen_strategy = st.selectbox("Strategy", ["short_term", "medium_term", "long_term"], index=1)
run = st.button("Run analysis", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    portfolio_df = pd.read_csv(uploaded)
    with st.spinner("Analyzing portfolio (only uncached tickers trigger new research)..."):
        result = analyze_portfolio(portfolio_df, chosen_strategy, risk_profile)

    if result["errors"]:
        st.warning("Portfolio validation issues (invalid rows were dropped):\n\n" + "\n".join(f"- {e}" for e in result["errors"]))

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
            with st.expander(f"{rec.ticker} -- {rec.action} (confidence {rec.confidence:.0%})"):
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
