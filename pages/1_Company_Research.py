"""Streamlit page: single-company research -- section 17.

Every section below is shown separately and stays visible -- fundamental,
technical, and institutional views are never collapsed into one opaque score
(section 21). Institutional context is an input, never the verdict (section
5). This page triggers NO new SEC/news LLM extraction (same cache-first
contract as `research_engine.load_company_research`) -- a ticker with nothing
cached yet simply shows thinner evidence, not an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging

import pandas as pd
import streamlit as st

import config
import horizon
import live_chart
import research_engine
import screener

# yfinance logs every unknown/delisted ticker at ERROR with a traceback; the
# screener already reports failures in the UI, so keep the console readable.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

st.set_page_config(page_title="Company Research", layout="wide")
st.title("Company Research")
st.caption(
    "A research view with stated evidence, confidence, and data freshness -- not personalized "
    "financial advice, not a promise of performance, and never a fabricated price target."
)

# --- Ticker + horizon (horizon carries over from the Universe page) -----------
default_ticker = st.session_state.get("selected_ticker", "")
ticker_input = st.text_input("Ticker", value=default_ticker, placeholder="e.g. AAPL").strip().upper()

horizon_label = st.select_slider(
    "Investment horizon",
    options=list(horizon.HORIZON_PRESETS.keys()),
    value=st.session_state.get("horizon_label", "1 year"),
)
st.session_state["horizon_label"] = horizon_label
horizon_days = horizon.HORIZON_PRESETS[horizon_label]
st.session_state["horizon_days"] = horizon_days

if not ticker_input:
    st.info("Enter a ticker, or click a company on the Opportunities page.")
    st.stop()

institutional_mentions = (
    pd.read_parquet(config.INSTITUTIONAL_MENTIONS_PATH) if config.INSTITUTIONAL_MENTIONS_PATH.exists() else pd.DataFrame()
)

with st.spinner(f"Loading research for {ticker_input}..."):
    try:
        extended = research_engine.load_extended_research(ticker_input, horizon_days, institutional_mentions=institutional_mentions)
    except Exception as exc:
        st.error(f"Could not load research for {ticker_input}: {exc}")
        st.stop()

views = research_engine.component_views(extended)
confidence = research_engine.research_confidence(views, extended["research"].get("signal_count", 0))

# --- Overview -------------------------------------------------------------
st.header("Overview")
fundamentals = extended["fundamentals"]
cols = st.columns(5)
cols[0].metric("Ticker", extended["ticker"])
cols[1].metric("Sector", fundamentals.get("sector") or "n/a")
cols[2].metric("Current price", f"${extended['current_price']:,.2f}" if extended["current_price"] is not None else "n/a")
cols[3].metric("Market cap", f"${fundamentals['market_cap']:,.0f}" if fundamentals.get("market_cap") else "n/a")
cols[4].metric("Horizon", horizon_label)

freshness = extended["data_freshness"]
if freshness["status"] == "STALE":
    st.warning(
        f"⚠️ Data freshness: **STALE** for a {horizon_label} horizon "
        f"(price: {freshness['last_price_update'] or 'n/a'}, "
        f"fundamentals: {freshness['last_fundamental_update'] or 'n/a'}, "
        f"institutional: {freshness['last_institutional_report_update'] or 'n/a'})."
    )
else:
    st.caption(f"Data freshness: OK for a {horizon_label} horizon.")

# --- Evidence verdict for this horizon (same scoring as the Opportunities ranking) ---
hv_top = extended["horizon_weighted_view"]
universe_path = config.INSTITUTIONAL_UNIVERSE_PATH
inst_score = 0.0
if universe_path.exists():
    uni = pd.read_parquet(universe_path)
    match = uni[uni["ticker"] == extended["ticker"]]
    if not match.empty and pd.notna(match.iloc[0].get("institutional_direction_score")):
        inst_score = float(match.iloc[0]["institutional_direction_score"])
rank_score = None if hv_top["score"] is None else hv_top["score"] + screener.INSTITUTIONAL_TILT * inst_score
verdict = screener.label_for(rank_score)
cached_ranking = screener.load_cached_ranking(horizon_days, universe_path)
rank_text = ""
if cached_ranking is not None and extended["ticker"] in set(cached_ranking["ticker"]):
    rank_text = f" -- ranked **#{int(cached_ranking.set_index('ticker').loc[extended['ticker'], 'rank'])} of {len(cached_ranking)}** in your universe"
verdict_text = (
    f"**Evidence for a {horizon_label} horizon: {verdict}**"
    + (f" (score {rank_score:+.2f})" if rank_score is not None else "")
    + rank_text
    + f". Confidence: {confidence['level']}. Not a return forecast; not backtested yet."
)
{"Strong": st.success, "Favorable": st.success, "Unfavorable": st.error}.get(verdict, st.info)(verdict_text)

# --- Live price chart -----------------------------------------------------------
st.header("Price chart")
default_window = live_chart.window_for_horizon(horizon_days)
window_names = list(live_chart.WINDOW_CHOICES)
default_name = next(
    (n for n, w in live_chart.WINDOW_CHOICES.items() if (w.period, w.interval) == (default_window.period, default_window.interval)),
    window_names[3],
)
ccols = st.columns([4, 1])
window_name = ccols[0].radio("Window", window_names, index=window_names.index(default_name), horizontal=True,
                             help="Defaults to the bars that matter for your horizon.")
window = live_chart.WINDOW_CHOICES[window_name]
auto_refresh = ccols[1].toggle("Live refresh", value=window.refresh_seconds is not None, disabled=window.refresh_seconds is None,
                               help="Re-fetches intraday bars automatically. Daily/weekly windows don't need it.")

_rr = extended["risk_reward"]
_mf = extended["research"].get("market_features", {}) or {}
chart_levels = {
    "Stop loss": _rr["stop_loss"].get("level"),
    "Take profit": _rr["take_profit"].get("level"),
    "Support": _mf.get("support_60d"),
    "Resistance": _mf.get("resistance_60d"),
}
show_levels = st.checkbox("Show stop / target / support / resistance", value=True)


@st.fragment(run_every=window.refresh_seconds if (auto_refresh and window.refresh_seconds) else None)
def render_price_chart() -> None:
    try:
        bars = live_chart.fetch_bars(ticker_input, window)
    except Exception as exc:
        st.warning(f"Could not fetch live bars for {ticker_input}: {exc}")
        return
    if bars.empty:
        st.info("No bars returned for this window (market closed, or no intraday data for this ticker).")
        return
    last, first = bars["close"].iloc[-1], bars["close"].iloc[0]
    mcols = st.columns(4)
    mcols[0].metric("Last", f"${last:,.2f}", f"{(last / bars['close'].iloc[-2] - 1):+.2%} vs prev bar" if len(bars) > 1 else None)
    mcols[1].metric("Window change", f"{(last / first - 1):+.1%}")
    mcols[2].metric("Window high / low", f"${bars['high'].max():,.2f} / ${bars['low'].min():,.2f}")
    mcols[3].metric("Last bar", pd.Timestamp(bars.index[-1]).strftime("%Y-%m-%d %H:%M"))
    fig = live_chart.build_figure(bars, ticker_input, chart_levels if show_levels else None, intraday=window.intraday)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"{window.description} · Yahoo Finance (intraday typically ~15 min delayed)"
        + (f" · auto-refreshing every {window.refresh_seconds}s" if auto_refresh and window.refresh_seconds else "")
        + ". SMAs and RSI are computed on the bars shown."
    )


render_price_chart()

st.divider()

# --- Institutional context --------------------------------------------------
st.header("Institutional context")
st.caption("Why this company is in the universe -- institutional input, not this project's verdict.")
if extended["institutional_context"].empty:
    st.caption("No institutional mentions found for this ticker.")
else:
    ctx = extended["institutional_context"]
    st.write(f"Mentioned by **{ctx['institution'].nunique()}** institution(s) across **{len(ctx)}** report reference(s).")
    display_cols = [c for c in ["institution", "report_title", "publication_date", "theme", "investment_horizon", "institutional_view", "view_direction", "confidence", "report_url"] if c in ctx.columns]
    st.dataframe(ctx[display_cols], width="stretch")

st.divider()

# --- Fundamental analysis ----------------------------------------------------
st.header("Fundamental analysis")


def _pct_metric(label: str, value, trend: str = None) -> None:
    text = f"{value:+.1%}" if value is not None else "n/a"
    if trend and trend != "INSUFFICIENT_DATA":
        text += f" ({trend.title()})"
    st.metric(label, text)


fcols = st.columns(4)
with fcols[0]:
    st.subheader("Growth")
    _pct_metric("Revenue growth", fundamentals.get("revenue_growth"), fundamentals.get("revenue_growth_trend"))
    _pct_metric("EPS growth", fundamentals.get("eps_growth"), fundamentals.get("eps_growth_trend"))
    _pct_metric("Operating income growth", fundamentals.get("operating_income_growth"))
    _pct_metric("FCF growth", fundamentals.get("fcf_growth"), fundamentals.get("fcf_trend"))
with fcols[1]:
    st.subheader("Profitability")
    _pct_metric("Gross margin", fundamentals.get("gross_margin"))
    _pct_metric("Operating margin", fundamentals.get("operating_margin"), fundamentals.get("operating_margin_trend"))
    _pct_metric("Net margin", fundamentals.get("net_margin"), fundamentals.get("net_margin_trend"))
    _pct_metric("ROE", fundamentals.get("roe"))
    _pct_metric("ROIC", fundamentals.get("roic"))
with fcols[2]:
    st.subheader("Balance sheet")
    st.metric("Cash", f"${fundamentals['cash']:,.0f}" if fundamentals.get("cash") is not None else "n/a")
    st.metric("Net debt", f"${fundamentals['net_debt']:,.0f}" if fundamentals.get("net_debt") is not None else "n/a")
    st.metric("Leverage (debt/equity)", f"{fundamentals['leverage']:.2f}x" if fundamentals.get("leverage") is not None else "n/a")
    st.metric("Interest coverage", f"{fundamentals['interest_coverage']:.1f}x" if fundamentals.get("interest_coverage") is not None else "n/a")
    st.caption(f"Balance sheet trend (cash-based): {fundamentals.get('balance_sheet_trend')}")
with fcols[3]:
    st.subheader("Valuation")
    st.metric("P/E (trailing)", f"{fundamentals['pe']:.1f}" if fundamentals.get("pe") is not None else "n/a")
    st.metric("Forward P/E", f"{fundamentals['forward_pe']:.1f}" if fundamentals.get("forward_pe") is not None else "n/a")
    st.metric("EV/EBITDA", f"{fundamentals['ev_ebitda']:.1f}" if fundamentals.get("ev_ebitda") is not None else "n/a")
    st.metric("Price/Book", f"{fundamentals['price_book']:.1f}" if fundamentals.get("price_book") is not None else "n/a")
    _pct_metric("FCF yield", fundamentals.get("fcf_yield"))
    st.caption(f"Valuation context (fwd vs trailing P/E): {fundamentals.get('valuation_context')}")

st.caption(
    f"Fundamentals as of {fundamentals.get('observation_date') or 'n/a'} "
    f"(retrieved {fundamentals.get('data_freshness') or 'n/a'}, sources: {', '.join(fundamentals.get('sources') or []) or 'n/a'}). "
    "Missing metrics reflect gaps in the free data source (yfinance), not zero values -- never fabricated."
)
if fundamentals.get("data_quality"):
    with st.expander(f"⚠️ {len(fundamentals['data_quality'])} data-quality flag(s) -- extreme/unusual values worth a second look, not deleted"):
        for flag in fundamentals["data_quality"]:
            st.write(f"- {flag}")

st.divider()

# --- Technical analysis -------------------------------------------------------
st.header("Technical analysis")
mf = extended["research"].get("market_features", {})
tcols = st.columns(4)
tcols[0].metric("1D / 5D / 20D return", f"{mf.get('return_1d', 0) or 0:+.1%} / {mf.get('return_5d', 0) or 0:+.1%} / {mf.get('return_20d', 0) or 0:+.1%}")
tcols[1].metric("6M / 1Y return", f"{mf.get('return_6m', 0) or 0:+.1%} / {mf.get('return_1y', 0) or 0:+.1%}")
tcols[2].metric("RSI(14)", f"{mf['rsi_14d']:.0f}" if mf.get("rsi_14d") is not None else "n/a")
tcols[3].metric("Volatility (1M, ann.)", f"{mf['volatility_1m']:.1%}" if mf.get("volatility_1m") is not None else "n/a")

tcols2 = st.columns(4)
tcols2[0].metric("Price vs MA50", f"{mf['price_vs_ma50']:+.1%}" if mf.get("price_vs_ma50") is not None else "n/a")
tcols2[1].metric("Price vs MA200", f"{mf['price_vs_ma200']:+.1%}" if mf.get("price_vs_ma200") is not None else "n/a")
tcols2[2].metric("Support (60D)", f"${mf['support_60d']:.2f}" if mf.get("support_60d") is not None else "n/a")
tcols2[3].metric("Resistance (60D)", f"${mf['resistance_60d']:.2f}" if mf.get("resistance_60d") is not None else "n/a")

st.divider()

# --- Historical evidence -------------------------------------------------------
st.header("Historical evidence")
research = extended["research"]
st.write(f"**{research['signal_count']}** underlying SEC/news signal(s), last dated {research.get('data_freshness') or 'n/a'}.")
ecols = st.columns(2)
with ecols[0]:
    st.write("**Key risks (from filings/news):**")
    for r in research.get("risks", []) or ["None identified."]:
        st.write(f"- {r}")
with ecols[1]:
    st.write("**Key positive catalysts:**")
    for c in research.get("catalysts", []) or ["None identified."]:
        st.write(f"- {c}")

st.divider()

# --- Risk / reward framework --------------------------------------------------
st.header("Risk / reward framework")
rr = extended["risk_reward"]
st.caption(rr["entry_context"])
rcols = st.columns(4)
rcols[0].metric("Stop loss", f"${rr['stop_loss']['level']:.2f}" if rr["stop_loss"]["level"] is not None else "n/a")
rcols[1].metric("Take profit", f"${rr['take_profit']['level']:.2f}" if rr["take_profit"]["level"] is not None else "n/a")
rcols[2].metric("Risk/reward ratio", f"{rr['risk_reward_ratio']:.2f}" if rr["risk_reward_ratio"] is not None else "n/a")
rcols[3].metric("Expected holding period", rr["expected_holding_horizon"])

st.write(f"**Stop-loss type:** `{rr['stop_loss']['type']}` -- {rr['stop_loss'].get('note') or rr['stop_loss'].get('method')}")
st.write(f"**Take-profit type:** `{rr['take_profit']['type']}` -- {rr['take_profit'].get('note') or rr['take_profit'].get('method')}")

st.divider()

# --- Horizon-aware weighting ----------------------------------------------------
st.header("Horizon-aware weighting")
hv = extended["horizon_weighted_view"]
st.caption(
    f"How much this {horizon_label} view leans on technicals vs fundamentals -- the same company can, "
    "and often should, weight these differently at a different horizon. This is descriptive weighting, "
    "not a recommendation."
)
hcols = st.columns(4)
hcols[0].metric("Horizon profile", hv["horizon_profile"])
hcols[1].metric("Technical weight", f"{hv['technical_weight']:.0%}")
hcols[2].metric("Fundamental weight", f"{hv['fundamental_weight']:.0%}")
hcols[3].metric("Weighted score", f"{hv['score']:+.2f}" if hv["score"] is not None else "n/a")

st.write("**Dominant factors at this horizon:** " + (", ".join(f.replace("_", " ") for f in hv["dominant_factors"]) or "none available"))

with st.expander("Component breakdown (score × weight = contribution, per group)"):
    if hv["components"]:
        component_rows = [
            {"component": group.replace("_", " "), "score": detail["score"], "weight": detail["weight"], "contribution": detail["contribution"]}
            for group, detail in sorted(hv["components"].items(), key=lambda kv: abs(kv[1]["contribution"]), reverse=True)
        ]
        st.dataframe(pd.DataFrame(component_rows), width="stretch")
    else:
        st.caption("No components had enough data at this horizon.")
    st.caption(
        "A missing component (e.g. no resolvable technicals) is excluded and the remaining weights "
        "renormalize -- it is never treated as a negative signal."
    )

st.divider()

# --- Research conclusion -------------------------------------------------------
st.header("Research conclusion")
vcols = st.columns(3)
vcols[0].metric("Fundamental", views["fundamental"])
vcols[1].metric("Technical", views["technical"])
vcols[2].metric("Institutional", views["institutional"])

st.write(
    f"**Horizon-weighted combined score ({horizon_label}):** "
    f"{views['horizon_fit_score']:.2f}" if views["horizon_fit_score"] is not None else "**Horizon-weighted combined score:** insufficient evidence"
)
st.write(f"**Confidence:** {confidence['level']}")
with st.expander("What drives this confidence level"):
    for driver in confidence["drivers"]:
        st.write(f"- {driver}")

agreement = {views["fundamental"], views["technical"], views["institutional"]} - {"INSUFFICIENT_EVIDENCE"}
if len(agreement) > 1:
    st.info(
        "The components disagree -- e.g. institutional research may be positive on a long-term theme "
        "while near-term technicals or fundamentals are weak. This is shown deliberately rather than "
        "averaged into one score: it means the case is more nuanced (or horizon-dependent) than a single "
        "label would suggest."
    )
elif agreement:
    st.success(f"Fundamental, technical, and institutional views that have evidence all point the same direction: {agreement.pop()}.")
else:
    st.warning("Insufficient evidence across all components for a research view at this horizon.")
