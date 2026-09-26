"""Streamlit front page: pick a horizon -> see the ranked opportunities.

Fully automatic flow:
1. Choose a horizon (1 day .. 20 years).
2. If the SEC 13F holdings are missing or older than a week, they're refreshed
   automatically (no files to drop in, no LLM).
3. Every company in the institutional universe is scored for that horizon
   with the same pipeline as the Company Research page (fundamentals +
   technicals + institutional activity, horizon-weighted) and ranked. Results
   are cached per horizon per day, so switching back is instant.
4. Click a row -> Company Research opens on that company, with live charts.

The labels (Strong / Favorable / Neutral / Unfavorable) describe where the
current evidence leans for the chosen horizon. They are not return forecasts,
and the method has not been backtested yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import logging

import pandas as pd
import streamlit as st

import config
import horizon
import price_provider
import screener
from institutional_research import documents, holdings_13f, parser, providers, universe

MAX_UI_CHUNKS_PER_RUN = 30  # safety cap for on-demand report ingestion triggered from the UI

# yfinance logs every unknown/delisted ticker at ERROR with a traceback; the
# screener already reports failures in the UI, so keep the console readable.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Opportunities", layout="wide")
st.title("Opportunities for your horizon")
st.caption(
    "Companies held and traded by the largest institutions (SEC 13F), ranked by this project's own "
    "fundamental + technical evidence for the horizon you pick. A ranking of evidence, not financial advice -- "
    "the scoring has not been backtested yet."
)

# --- Horizon ------------------------------------------------------------------
horizon_label = st.select_slider(
    "Holding period",
    options=list(horizon.HORIZON_PRESETS.keys()),
    value=st.session_state.get("horizon_label", "6 months"),
)
horizon_days = horizon.HORIZON_PRESETS[horizon_label]
st.session_state["horizon_label"] = horizon_label
st.session_state["horizon_days"] = horizon_days


def refresh_13f_and_universe(filers: dict) -> list[dict]:
    new_13f, summaries = holdings_13f.refresh_13f_mentions(filers)
    combined = holdings_13f.merge_13f_mentions(new_13f, config.INSTITUTIONAL_MENTIONS_PATH)
    holdings_13f.mark_refreshed()
    universe.build_universe(combined).to_parquet(config.INSTITUTIONAL_UNIVERSE_PATH, index=False)
    return summaries


# --- Automatic data refresh ------------------------------------------------------
filers = holdings_13f.load_filers()
if holdings_13f.needs_refresh() and not st.session_state.get("auto_13f_attempted"):
    st.session_state["auto_13f_attempted"] = True  # once per session, even if SEC is down
    with st.status("Updating institutional holdings from SEC 13F filings (first run: a few minutes)...", expanded=False) as status:
        summaries = refresh_13f_and_universe(filers)
        failed = [s for s in summaries if not str(s["status"]).startswith("ok")]
        status.update(label=f"13F holdings updated ({len(summaries) - len(failed)}/{len(summaries)} institutions)", state="error" if failed else "complete")
        st.dataframe(pd.DataFrame(summaries), width="stretch")

mentions = pd.read_parquet(config.INSTITUTIONAL_MENTIONS_PATH) if config.INSTITUTIONAL_MENTIONS_PATH.exists() else pd.DataFrame()
universe_df = pd.read_parquet(config.INSTITUTIONAL_UNIVERSE_PATH) if config.INSTITUTIONAL_UNIVERSE_PATH.exists() else pd.DataFrame()

if universe_df.empty:
    st.error(
        "No universe yet -- the automatic SEC 13F refresh didn't produce any companies. "
        "Open **Data sources** below to retry and see per-institution errors."
    )
else:
    # --- Ranking (cached per horizon per day) -----------------------------------
    ranking = screener.load_cached_ranking(horizon_days, config.INSTITUTIONAL_UNIVERSE_PATH)
    if ranking is None or st.session_state.pop("force_rerank", False):
        bar = st.progress(0.0, text=f"Scoring {len(universe_df)} companies for a {horizon_label} horizon...")
        ranking = screener.rank_universe(
            universe_df, horizon_days, mentions,
            prefetch=price_provider.get_price_history,
            progress=lambda done, total, t: bar.progress(done / total, text=f"Scoring for {horizon_label}: {t} ({done}/{total})"),
        )
        bar.empty()
        screener.save_ranking(ranking, horizon_days, config.INSTITUTIONAL_UNIVERSE_PATH)

    # --- Filters ------------------------------------------------------------------
    fcols = st.columns([2, 2, 2, 1])
    sector_filter = fcols[0].multiselect("Sector", sorted(ranking["sector"].dropna().unique().tolist()))
    institution_options = sorted(mentions["institution"].dropna().unique().tolist()) if not mentions.empty else []
    institution_filter = fcols[1].multiselect("Held / traded by", institution_options)
    show = fcols[2].radio("Show", ["Shortlist", "All ranked"], horizontal=True)
    if fcols[3].button("Re-score", help="Recompute the ranking now (prices/fundamentals are cached ~20h)"):
        st.session_state["force_rerank"] = True
        st.rerun()

    view = ranking
    if show == "Shortlist":
        view = view[view["shortlisted"]]
    if sector_filter:
        view = view[view["sector"].isin(sector_filter)]
    if institution_filter:
        view = view[view["ticker"].isin(set(mentions[mentions["institution"].isin(institution_filter)]["ticker"].dropna()))]

    n_short = int(ranking["shortlisted"].sum())
    st.subheader(f"{len(view)} compan{'y' if len(view) == 1 else 'ies'} -- {horizon_label} horizon")
    st.caption(
        f"Shortlist = label Strong/Favorable, fresh data, and enough evidence ({n_short} of {len(ranking)} companies today). "
        f"Score = horizon-weighted fundamentals/technicals (−1..+1) + {screener.INSTITUTIONAL_TILT:.2f} × institutional direction. "
        "**Click a row** to open the full analysis with live charts."
    )

    table_cols = ["rank", "ticker", "company_name", "sector", "label", "rank_score", "technical_view", "fundamental_view",
                  "institutional_view", "confidence", "current_price", "return_20d", "stop_loss", "take_profit",
                  "risk_reward_ratio", "dominant_factors"]
    event = st.dataframe(
        view[table_cols].reset_index(drop=True),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"ranking_{horizon_days}_{show}",
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "company_name": "Company",
            "rank_score": st.column_config.ProgressColumn("Score", min_value=-1.0, max_value=1.0, format="%.2f"),
            "technical_view": "Technical",
            "fundamental_view": "Fundamental",
            "institutional_view": "Institutions",
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "return_20d": st.column_config.NumberColumn("20D", format="percent"),
            "stop_loss": st.column_config.NumberColumn("Stop", format="$%.2f"),
            "take_profit": st.column_config.NumberColumn("Target", format="$%.2f"),
            "risk_reward_ratio": st.column_config.NumberColumn("R/R", format="%.2f"),
            "dominant_factors": "Driven by",
        },
    )
    if event.selection.rows:
        st.session_state["selected_ticker"] = view.iloc[event.selection.rows[0]]["ticker"]
        st.switch_page("pages/1_Company_Research.py")

    if show == "Shortlist" and view.empty:
        st.info("Nothing passes the shortlist for this horizon right now -- switch to **All ranked** to see every company's score.")

    failed = ranking[ranking["error"].notna()]
    if not failed.empty:
        with st.expander(f"{len(failed)} compan(ies) could not be scored"):
            st.dataframe(failed[["ticker", "company_name", "error"]], width="stretch", hide_index=True)

st.divider()

# --- Data sources (automatic by default; manual controls here) ---------------------
with st.expander("Data sources"):
    st.markdown(
        f"**SEC 13F holdings** -- {', '.join(filers)}. Refreshed automatically when older than "
        f"{holdings_13f.AUTO_REFRESH_MAX_AGE_DAYS} days (last: {holdings_13f.last_refreshed_at() or 'never'}). "
        "Edit `data/raw/13f_filers.csv` (`institution,cik`) to change the list. 13Fs are filed up to 45 days after "
        "quarter end; adds/cuts are measured relative to each filer's median change."
    )
    if st.button("Refresh 13F holdings now"):
        with st.spinner("Fetching latest 13F filings from SEC EDGAR..."):
            summaries = refresh_13f_and_universe(filers)
        st.dataframe(pd.DataFrame(summaries), width="stretch")
        st.rerun()

    if not mentions.empty:
        st.markdown("**Institutional themes** (from ingested reports, independent of whether a company was identified)")
        st.dataframe(universe.theme_summary(mentions), width="stretch")

    st.markdown("**Optional: institutional research reports** (LLM-parsed outlooks, adds themes and extra mentions)")
    st.caption(
        f"Drop report files (.txt, .md, .pdf, .html) into `data/raw/institutional/<institution>/` "
        f"(e.g. `data/raw/institutional/BlackRock/2026_outlook.pdf`), then parse them below. "
        f"An optional `<file>.meta.json` sidecar can state the real title/date/URL exactly."
    )

    loaded_reports = documents.load_all_reports()
    st.write(f"**{len(loaded_reports)}** report file(s) found under `{config.INSTITUTIONAL_RAW_DIR}`.")

    col1, col2 = st.columns(2)
    with col1:
        fetch_urls = st.checkbox(
            "Also fetch reports from a URL list first (network, opt-in)",
            value=False,
            help=f"Downloads every not-yet-fetched URL in {config.INSTITUTIONAL_URL_LIST_PATH} "
            "(columns: institution, report_title, url[, report_type, publication_date]). "
            "Checks robots.txt and rate-limits per host. Never runs automatically.",
        )
    with col2:
        run_ingestion = st.button("Parse reports into the universe", type="primary", disabled=not loaded_reports and not fetch_urls)

    if run_ingestion:
        if fetch_urls:
            with st.spinner("Fetching reports from the URL list (network)..."):
                fetch_results = providers.fetch_from_url_list()
            st.dataframe(pd.DataFrame(fetch_results))
            loaded_reports = documents.load_all_reports()

        with st.spinner(f"Extracting institutional mentions (cache-first, up to {MAX_UI_CHUNKS_PER_RUN} new LLM calls)..."):
            new_mentions, summary = parser.run_extraction(loaded_reports, max_chunks=MAX_UI_CHUNKS_PER_RUN)
            combined_mentions = parser.save_mentions(new_mentions, config.INSTITUTIONAL_MENTIONS_PATH)
            universe_df = universe.build_universe(combined_mentions)
            universe_df.to_parquet(config.INSTITUTIONAL_UNIVERSE_PATH, index=False)

        st.success(
            f"Processed {summary['chunks_processed']} chunk(s), {summary['llm_calls_made']} new LLM call(s) "
            f"(rest served from cache) -> {summary['mentions_extracted']} new mention(s)."
        )
        if summary["llm_calls_skipped_over_limit"]:
            st.warning(
                f"{summary['llm_calls_skipped_over_limit']} chunk(s) were skipped this run "
                "(MAX_UI_CHUNKS_PER_RUN reached) -- run again to continue them."
            )
