"""Streamlit page: Quick Picks -- the ranking reduced to BUY / SELL lines.

For when you don't want to read the analysis: pick a horizon and a budget,
get what to buy (at what price, with what stop and target, how many shares)
and what to sell or avoid. Same ranking as the Opportunities page (cached
per horizon per day), just presented as an order list.

The sizing rule is fixed and visible: risk at most X% of the capital per
trade if the stop is hit, never more than 20% of the capital in one line.
It is the current evidence turned into an order list, not a performance
promise: the point-in-time backtest (README STEP 11b) found no statistically
significant out-of-sample edge in this ranking, and the page says so up top.
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

BACKTEST_README_URL = "https://github.com/AntoBrgt/Quant-research-lab#step-11b----point-in-time-13f-universe-for-the-backtest"

st.set_page_config(page_title="Quick Picks", layout="wide")
st.title("Quick Picks")
st.warning(
    "**Backtest (STEP 11b): this ranking has shown no statistically significant edge out of sample. "
    "Treat as a watchlist, not trade instructions.** "
    f"[Details in the README]({BACKTEST_README_URL})"
)
st.caption("What to buy, at what price, how much -- and what to sell. The short version of the Opportunities ranking.")

universe_df = pd.read_parquet(config.INSTITUTIONAL_UNIVERSE_PATH) if config.INSTITUTIONAL_UNIVERSE_PATH.exists() else pd.DataFrame()
mentions = pd.read_parquet(config.INSTITUTIONAL_MENTIONS_PATH) if config.INSTITUTIONAL_MENTIONS_PATH.exists() else pd.DataFrame()
if universe_df.empty:
    st.info("No universe yet -- open the **Opportunities** page once; it fetches the institutional data automatically.")
    st.stop()

# --- Inputs ----------------------------------------------------------------------
c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1])
horizon_label = c1.select_slider(
    "Holding period", options=list(horizon.HORIZON_PRESETS.keys()), value=st.session_state.get("horizon_label", "6 months")
)
horizon_days = horizon.HORIZON_PRESETS[horizon_label]
st.session_state["horizon_label"] = horizon_label
st.session_state["horizon_days"] = horizon_days
capital = c2.number_input("Budget ($)", min_value=0.0, value=float(st.session_state.get("qp_capital", 10_000.0)), step=500.0)
st.session_state["qp_capital"] = capital
risk_pct = c3.number_input("Max loss per trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                           help="If a stop is hit, you lose at most this share of your budget on that trade.") / 100
n_picks = c4.number_input("Picks", min_value=1, max_value=25, value=5)

bar = st.empty()
ranking = screener.get_ranking(
    universe_df, horizon_days, mentions, config.INSTITUTIONAL_UNIVERSE_PATH,
    progress=lambda done, total, t: bar.progress(done / total, text=f"Scoring for {horizon_label}: {t} ({done}/{total})"),
)
bar.empty()
buys, sells = screener.quick_picks(ranking, n_buy=int(n_picks), n_sell=int(n_picks), capital=capital, risk_pct=risk_pct)

price_cols = {
    "ticker": "Ticker",
    "company_name": "Company",
    "price": st.column_config.NumberColumn("Price", format="$%.2f"),
    "stop_loss": st.column_config.NumberColumn("Stop", format="$%.2f"),
    "take_profit": st.column_config.NumberColumn("Target", format="$%.2f"),
    "upside": st.column_config.NumberColumn("To target", format="percent"),
    "downside": st.column_config.NumberColumn("To stop", format="percent"),
    "risk_reward_ratio": st.column_config.NumberColumn("R/R", format="%.2f"),
    "shares": st.column_config.NumberColumn("Shares", format="%d"),
    "amount": st.column_config.NumberColumn("Amount", format="$%.0f"),
    "why": "Why",
}


def _money(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"${value:,.2f}"


# --- BUY ---------------------------------------------------------------------------
st.header(f"Buy -- {horizon_label}")
if buys.empty:
    st.info("Nothing clears the bar for this horizon today. Doing nothing is a valid answer; try another horizon.")
else:
    for _, row in buys.head(3).iterrows():
        size = f"**{int(row['shares'])} shares** (~${row['amount']:,.0f})" if row["shares"] else "size: n/a"
        exit_plan = (
            f"stop **{_money(row['stop_loss'])}** ({row['downside']:+.1%}), target **{_money(row['take_profit'])}** ({row['upside']:+.1%})"
            if pd.notna(row["stop_loss"]) and pd.notna(row["take_profit"])
            else "no fixed stop/target at this horizon -- exit if the thesis breaks (see Company Research)"
        )
        st.success(f"**BUY {row['ticker']}** ({row['company_name']}) at **≤ {_money(row['price'])}** · {size} · {exit_plan}")

    invested = buys["amount"].fillna(0).sum()
    st.caption(
        f"Total: ~${invested:,.0f} of ${capital:,.0f} budget. Sizing: lose at most {risk_pct:.1%} of the budget per trade if the stop is hit, "
        f"max {screener.MAX_POSITION_PCT:.0%} of the budget in one line. Only names whose target is at least as far as their stop (R/R ≥ 1) are listed. Use a limit order at the price shown and place the stop right after the fill."
    )
    event = st.dataframe(
        buys[list(price_cols) + ["sizing", "confidence"]], hide_index=True, width="stretch",
        column_config=price_cols, on_select="rerun", selection_mode="single-row", key=f"qp_buy_{horizon_days}",
    )
    if event.selection.rows:
        st.session_state["selected_ticker"] = buys.iloc[event.selection.rows[0]]["ticker"]
        st.switch_page("pages/1_Company_Research.py")

# --- SELL --------------------------------------------------------------------------
st.header(f"Sell / avoid -- {horizon_label}")
if sells.empty:
    st.caption("No company is currently Unfavorable for this horizon.")
else:
    st.caption("If you hold one of these, the evidence for this horizon points the wrong way: sell at market or at least tighten your stop to the level shown. If you don't, don't buy it.")
    for _, row in sells.head(3).iterrows():
        tighten = f" -- or keep a stop at **{_money(row['stop_loss'])}**" if pd.notna(row["stop_loss"]) else ""
        st.error(f"**SELL {row['ticker']}** ({row['company_name']}) around **{_money(row['price'])}**{tighten}")
    sell_cols = {k: v for k, v in price_cols.items() if k not in ("shares", "amount", "take_profit", "upside")}
    event = st.dataframe(
        sells[list(sell_cols) + ["confidence"]], hide_index=True, width="stretch",
        column_config=sell_cols, on_select="rerun", selection_mode="single-row", key=f"qp_sell_{horizon_days}",
    )
    if event.selection.rows:
        st.session_state["selected_ticker"] = sells.iloc[event.selection.rows[0]]["ticker"]
        st.switch_page("pages/1_Company_Research.py")

st.divider()
st.caption(
    f"Based on today's ranking of {len(ranking)} companies (prices ~20h cache; Yahoo data). Labels describe current evidence, "
    "not a return forecast, and the method has not been backtested yet. Not financial advice -- you place the orders."
)
