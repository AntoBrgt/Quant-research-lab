"""Streamlit page: Quick Picks -- a WATCHLIST, not an order list.

PHASE 2 (PREREGISTRATION_PHASE2.md) tested three rules that could have put BUY
lines on this page; none passed (README, PHASE 2). Its "No pass" clause applies:
no BUY/SELL orders, no position sizing, no stop/target instructions. The page
shows the same ranking as the Opportunities page, reduced to two short lists to
research, under the label "no validated strategy".
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st

import config
import horizon
import screener

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BACKTEST_README_URL = "https://github.com/AntoBrgt/Quant-research-lab/blob/main/docs/RESEARCH_LOG.md#phase-2----three-pre-registered-event-rules-python-srcrun_phase2py"

st.set_page_config(page_title="Quick Picks", layout="wide")
st.title("Quick Picks -- watchlist")
st.warning(
    "**No validated strategy.** Every ranking, trading rule and model tested on point-in-time data after costs "
    "(PHASE 1, 1b and 2) failed to beat simply holding SPY or IWM. This page is a watchlist to research, "
    "not buy or sell instructions. "
    f"[Details in the README]({BACKTEST_README_URL})"
)
st.caption("The top and bottom of the Opportunities ranking for one horizon -- names to look at, nothing more.")

universe_df = pd.read_parquet(config.INSTITUTIONAL_UNIVERSE_PATH) if config.INSTITUTIONAL_UNIVERSE_PATH.exists() else pd.DataFrame()
mentions = pd.read_parquet(config.INSTITUTIONAL_MENTIONS_PATH) if config.INSTITUTIONAL_MENTIONS_PATH.exists() else pd.DataFrame()
if universe_df.empty:
    st.info("No universe yet -- open the **Opportunities** page once; it fetches the institutional data automatically.")
    st.stop()

# --- Inputs ----------------------------------------------------------------------
c1, c2 = st.columns([3, 1])
horizon_label = c1.select_slider(
    "Horizon", options=list(horizon.HORIZON_PRESETS.keys()), value=st.session_state.get("horizon_label", "6 months")
)
horizon_days = horizon.HORIZON_PRESETS[horizon_label]
st.session_state["horizon_label"] = horizon_label
st.session_state["horizon_days"] = horizon_days
n_names = c2.number_input("Names per list", min_value=1, max_value=25, value=5)

bar = st.empty()
ranking = screener.get_ranking(
    universe_df, horizon_days, mentions, config.INSTITUTIONAL_UNIVERSE_PATH,
    progress=lambda done, total, t: bar.progress(done / total, text=f"Scoring for {horizon_label}: {t} ({done}/{total})"),
)
bar.empty()
top, bottom = screener.quick_picks(ranking, n_buy=int(n_names), n_sell=int(n_names))

cols = {
    "ticker": "Ticker",
    "company_name": "Company",
    "label": "Evidence label",
    "price": st.column_config.NumberColumn("Last price", format="$%.2f"),
    "why": "Why it ranks here",
    "confidence": "Confidence",
}


def _table(frame: pd.DataFrame, key: str) -> None:
    event = st.dataframe(frame[[c for c in cols if c in frame.columns]], hide_index=True, width="stretch",
                         column_config=cols, on_select="rerun", selection_mode="single-row", key=key)
    if event.selection.rows:
        st.session_state["selected_ticker"] = frame.iloc[event.selection.rows[0]]["ticker"]
        st.switch_page("pages/1_Company_Research.py")


st.header(f"Top of the ranking -- {horizon_label}")
if top.empty:
    st.info("No name clears the shortlist bar for this horizon today.")
else:
    _table(top, f"qp_top_{horizon_days}")

st.header(f"Bottom of the ranking -- {horizon_label}")
if bottom.empty:
    st.caption("No company is currently Unfavorable for this horizon.")
else:
    _table(bottom, f"qp_bottom_{horizon_days}")

st.divider()
st.caption(
    f"Based on today's ranking of {len(ranking)} companies (prices ~20h cache; Yahoo data). Labels describe current evidence, "
    "not a return forecast. The ranking was backtested and showed no edge (README, PHASE 1 VERDICT). Not financial advice."
)
