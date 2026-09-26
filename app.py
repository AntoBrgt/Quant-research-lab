"""Streamlit front page: institutional universe -- section 16, steps 1-4.

This is now the primary flow (moved from a secondary page) -- portfolio
upload lives at pages/2_Portfolio_Upload.py for anyone who still wants it.

This page only ever answers "why is this company in the universe" (which
institutions mention it, which themes, what direction). It never computes or
shows a BUY/SELL/HOLD verdict -- that's the Company Research page, and even
there the institutional view stays one visible input, not the final word.

Ingestion (parsing local reports, or fetching a user-supplied URL list) is
explicit, opt-in, and cache-first here -- nothing on this page triggers LLM
calls or network access just by being viewed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

import config
import horizon
from institutional_research import documents, holdings_13f, parser, providers, universe

MAX_UI_CHUNKS_PER_RUN = 30  # safety cap for on-demand ingestion triggered from the UI

st.set_page_config(page_title="Institutional Universe", layout="wide")
st.title("Institutional Universe")
st.caption(
    "Companies, sectors, and themes surfaced from institutional research reports. "
    "This is institutional **context**, not a recommendation -- the project's own "
    "research (fundamentals, technicals, historical evidence) is separate and lives "
    "on the Company Research page."
)

# --- 1. Investment horizon (carries across pages via session_state) -----------
st.header("1. Investment horizon")
horizon_label = st.select_slider(
    "How long is your intended holding period?",
    options=list(horizon.HORIZON_PRESETS.keys()),
    value=st.session_state.get("horizon_label", "1 year"),
)
st.session_state["horizon_label"] = horizon_label
st.session_state["horizon_days"] = horizon.HORIZON_PRESETS[horizon_label]
st.caption(f"Selected horizon: **{horizon_label}** ({st.session_state['horizon_days']} days). This carries over to Company Research.")

st.divider()

# --- 2. Automatic source: SEC 13F holdings -------------------------------------
st.header("2. Institutional holdings (SEC 13F, automatic)")
filers = holdings_13f.load_filers()
st.caption(
    "Actual quarterly US-equity positions filed with the SEC by: **"
    + ", ".join(filers)
    + "**. No LLM, no manual files. Edit `data/raw/13f_filers.csv` (columns `institution, cik`) to change the list. "
    "13Fs are filed up to 45 days after quarter end, long positions only -- a holding or an add is context, not a buy signal. "
    "Adds/cuts are measured relative to each filer's median change, so index-fund inflows don't look like conviction."
)
last_13f = holdings_13f.last_refreshed_at()
st.write(f"Last 13F refresh: **{last_13f or 'never'}**")
if st.button("Refresh 13F holdings from SEC", type="primary" if not last_13f else "secondary"):
    with st.spinner("Fetching latest 13F filings from SEC EDGAR, resolving tickers, rebuilding the universe (1-3 min)..."):
        new_13f, summaries_13f = holdings_13f.refresh_13f_mentions(filers)
        combined_mentions = holdings_13f.merge_13f_mentions(new_13f, config.INSTITUTIONAL_MENTIONS_PATH)
        holdings_13f.mark_refreshed()
        universe_df = universe.build_universe(combined_mentions)
        universe_df.to_parquet(config.INSTITUTIONAL_UNIVERSE_PATH, index=False)
    st.dataframe(pd.DataFrame(summaries_13f), use_container_width=True)
    if any(not str(s["status"]).startswith("ok") for s in summaries_13f):
        st.warning("Some filers failed -- see the status column. The others were still saved.")
    else:
        st.success(f"{len(new_13f)} holding rows from {len(summaries_13f)} filers -> {len(universe_df)} companies in the universe.")

st.divider()

# --- 2b. Optional: research reports (explicit, opt-in, cache-first) -----------
st.header("2b. Optional: ingest institutional research reports")
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

st.divider()

# --- 3 & 4. Explore + filter the universe --------------------------------------
st.header("3. Explore the universe")

mentions = pd.read_parquet(config.INSTITUTIONAL_MENTIONS_PATH) if config.INSTITUTIONAL_MENTIONS_PATH.exists() else pd.DataFrame()
universe_df = pd.read_parquet(config.INSTITUTIONAL_UNIVERSE_PATH) if config.INSTITUTIONAL_UNIVERSE_PATH.exists() else pd.DataFrame()

if universe_df.empty:
    st.info(
        "No universe built yet. Click **Refresh 13F holdings from SEC** above (automatic), and/or ingest "
        "institutional reports -- the universe does not seed itself from any other tickers in this project."
    )
    st.stop()

with st.expander("Institutional themes (independent of whether a specific company was identified)"):
    st.dataframe(universe.theme_summary(mentions), use_container_width=True)

filter_cols = st.columns(4)
region_filter = filter_cols[0].multiselect("Region", sorted(universe_df["region"].dropna().unique().tolist()))
sector_filter = filter_cols[1].multiselect("Sector", sorted(universe_df["sector"].dropna().unique().tolist()))
all_themes = sorted({t for themes in universe_df["themes"] for t in themes})
theme_filter = filter_cols[2].multiselect("Theme", all_themes)
direction_filter = filter_cols[3].multiselect("Institutional direction", sorted(universe_df["institutional_direction"].dropna().unique().tolist()))
institution_options = sorted(mentions["institution"].dropna().unique().tolist()) if not mentions.empty else []
institution_filter = st.multiselect("Held / mentioned by institution", institution_options)

max_mentions = int(universe_df["institutional_mentions"].max())
if max_mentions > 1:
    min_mentions = st.slider("Minimum institutional mentions", 1, max_mentions, 1)
else:
    min_mentions = 1  # st.slider requires min < max; nothing to filter on with only one mention count present

filtered = universe_df[universe_df["institutional_mentions"] >= min_mentions]
if region_filter:
    filtered = filtered[filtered["region"].isin(region_filter)]
if sector_filter:
    filtered = filtered[filtered["sector"].isin(sector_filter)]
if theme_filter:
    filtered = filtered[filtered["themes"].apply(lambda themes: any(t in themes for t in theme_filter))]
if direction_filter:
    filtered = filtered[filtered["institutional_direction"].isin(direction_filter)]
if institution_filter:
    tickers_for_institutions = set(mentions[mentions["institution"].isin(institution_filter)]["ticker"].dropna())
    filtered = filtered[filtered["ticker"].isin(tickers_for_institutions)]

st.write(f"**{len(filtered)}** compan(ies) match the current filters.")
st.dataframe(
    filtered[["ticker", "company_name", "region", "sector", "market_cap", "themes", "institution_count", "institutional_mentions", "institutional_direction", "latest_mention_date"]],
    use_container_width=True,
)

st.divider()
st.header("4. Select a company")
if filtered.empty:
    st.caption("No companies match the current filters.")
else:
    selected = st.selectbox("Ticker", filtered["ticker"].tolist())
    st.session_state["selected_ticker"] = selected
    # A direct st.page_link to another page is fragile across Streamlit
    # versions/entrypoints (it resolves paths relative to the app's actual
    # entrypoint, which differs between a real `streamlit run app.py` session
    # and this page tested/opened in isolation) -- session_state is the
    # reliable hand-off, and the sidebar nav (always present in a multipage
    # app) is the reliable way to actually navigate.
    st.success(f"**{selected}** selected -- open **Company Research** from the sidebar to see its full research view.")
