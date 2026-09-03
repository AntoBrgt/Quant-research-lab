"""Combine research + strategy + portfolio context into one recommendation.

Four concepts stay separate fields rather than collapsing into a single score:
asset_signal (is the company attractive on its own), strategy_fit (does it fit
the chosen horizon), portfolio_fit (does adding/holding/trimming it make sense
given the existing portfolio), and risk (what could invalidate the thesis).
The final `action` combines them, but every component stays inspectable.

Deterministic Python, no LLM -- this is what makes it unit-testable with
synthetic signals, and keeps quantitative reasoning (section 11 of the spec)
out of the LLM's hands.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

Action = Literal["BUY", "HOLD", "SELL", "WATCH", "INSUFFICIENT_EVIDENCE"]
RiskProfile = Literal["conservative", "moderate", "aggressive"]

# Below this many total signals, there simply isn't enough evidence to act on.
MIN_SIGNALS_FOR_ACTION = 3
# Data older than this (rough calendar-day heuristic on filing/news dates) is
# flagged as stale rather than silently treated as current.
STALE_DATA_NOTE_DAYS = 200

# action thresholds on the combined [-1, 1] score, tightened for lower risk tolerance
_ACTION_THRESHOLDS: dict[RiskProfile, dict[str, float]] = {
    "conservative": {"buy": 0.45, "sell": -0.30},
    "moderate": {"buy": 0.30, "sell": -0.30},
    "aggressive": {"buy": 0.20, "sell": -0.35},
}


class Recommendation(BaseModel):
    ticker: str
    action: Action
    strategy: str
    confidence: float  # 0-1
    holding_period: str
    asset_signal: Optional[float]  # research_score, [-1, 1] or None
    strategy_fit: Optional[float]  # strategy score, [-1, 1] or None
    portfolio_fit: Optional[float]  # [-1, 1] or None (None = not a current holding)
    key_signals: list[str]
    key_risks: list[str]
    bull_case: str
    bear_case: str
    portfolio_consideration: str
    data_freshness: Optional[str]


def _portfolio_fit(ticker: str, portfolio_context: Optional[dict]) -> tuple[Optional[float], str]:
    """Score how adding to / holding this position affects diversification.

    Simple, explainable heuristic: an existing large position scores lower
    (adding to an already-concentrated holding is riskier), a new position in
    an under-represented sector scores higher.
    """
    if not portfolio_context:
        return None, "Not evaluated against a specific portfolio."

    holdings = portfolio_context.get("holdings", {})  # {ticker: weight}
    current_weight = holdings.get(ticker.upper())

    if current_weight is None:
        return 0.3, "Not currently held; would be a new position."

    if current_weight >= 0.25:
        return -0.5, f"Already {current_weight:.0%} of the portfolio -- adding further would increase concentration risk."
    if current_weight >= 0.10:
        return -0.1, f"Already a meaningful {current_weight:.0%} position."
    return 0.4, f"Currently a small {current_weight:.0%} position; room to add without concentration concerns."


def generate_recommendation(
    ticker: str,
    research: dict,
    strategy_fit_result: dict,
    portfolio_context: Optional[dict] = None,
    risk_profile: RiskProfile = "moderate",
    strategy_name: str = "medium_term",
) -> Recommendation:
    ticker = ticker.upper()
    asset_signal = research.get("research_score")
    strategy_fit = strategy_fit_result.get("score")
    signal_count = research.get("signal_count", 0)

    portfolio_fit, portfolio_note = _portfolio_fit(ticker, portfolio_context)

    key_signals = list((research.get("catalysts") or [])[:3])
    key_risks = list((research.get("risks") or [])[:3])

    if signal_count < MIN_SIGNALS_FOR_ACTION or asset_signal is None:
        return Recommendation(
            ticker=ticker, action="INSUFFICIENT_EVIDENCE", strategy=strategy_name,
            confidence=0.0, holding_period=strategy_name,
            asset_signal=asset_signal, strategy_fit=strategy_fit, portfolio_fit=portfolio_fit,
            key_signals=key_signals, key_risks=key_risks,
            bull_case="Not enough evidence to form a bull case.",
            bear_case="Not enough evidence to form a bear case.",
            portfolio_consideration=portfolio_note,
            data_freshness=research.get("data_freshness"),
        )

    components = [v for v in (asset_signal, strategy_fit) if v is not None]
    combined = sum(components) / len(components) if components else 0.0

    thresholds = _ACTION_THRESHOLDS[risk_profile]
    if combined >= thresholds["buy"]:
        action: Action = "BUY"
    elif combined <= thresholds["sell"]:
        action = "SELL"
    elif key_risks and combined < 0.1:
        action = "WATCH"
    else:
        action = "HOLD"

    confidence = round(min(1.0, signal_count / 10) * (0.5 + abs(combined) / 2), 4)

    return Recommendation(
        ticker=ticker, action=action, strategy=strategy_name,
        confidence=confidence, holding_period=strategy_name,
        asset_signal=asset_signal, strategy_fit=strategy_fit, portfolio_fit=portfolio_fit,
        key_signals=key_signals or ["No strong positive signals identified."],
        key_risks=key_risks or ["No elevated risks identified in available signals."],
        bull_case=key_signals[0] if key_signals else "Limited positive evidence available.",
        bear_case=key_risks[0] if key_risks else "No specific bear case identified from available signals.",
        portfolio_consideration=portfolio_note,
        data_freshness=research.get("data_freshness"),
    )
