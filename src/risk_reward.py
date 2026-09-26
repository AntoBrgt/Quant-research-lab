"""Horizon-dependent risk/reward framework. Deterministic Python, no LLM.

The brief is explicit that a tight technical stop and a "thesis invalidation"
level are different concepts, and that forcing one onto every horizon is
wrong: a 2%-below-support stop makes sense for a 1-week trade and is noise for
a 10-year holding, while "exit if the growth thesis breaks" is meaningless for
a 1-day trade and is exactly the right frame for a 10-year one. This module
picks the appropriate one from `horizon.horizon_category()` rather than
applying one universal formula, and never invents a numeric price target for
the long-horizon case -- consistent with this project's existing "never a
fabricated price target" boundary (see `recommendations.py`).
"""

from __future__ import annotations

from typing import Optional

import horizon as horizon_mod

# ATR multiples for a technical stop -- documented heuristic defaults (not
# fit to any backtest), tunable per horizon category.
ATR_STOP_MULTIPLE = {"very_short_term": 1.5, "short_term": 2.5, "medium_term": 3.0}
# Reward:risk multiple used for a technical take-profit target when no
# resistance level is available.
DEFAULT_REWARD_RISK_MULTIPLE = 2.0


def _format_horizon(horizon_days: int) -> str:
    if horizon_days < 30:
        return f"{horizon_days} day{'s' if horizon_days != 1 else ''}"
    if horizon_days < 365:
        months = round(horizon_days / 30)
        return f"~{months} month{'s' if months != 1 else ''}"
    years = round(horizon_days / 365, 1)
    years_str = str(int(years)) if years == int(years) else str(years)
    return f"~{years_str} year{'s' if years != 1 else ''}"


def _entry_context(current_price: Optional[float], market_features: dict) -> str:
    if current_price is None:
        return "No current price available -- entry context cannot be assessed."

    parts = []
    price_vs_ma50 = market_features.get("price_vs_ma50")
    if price_vs_ma50 is not None:
        direction = "above" if price_vs_ma50 >= 0 else "below"
        parts.append(f"price is {abs(price_vs_ma50):.1%} {direction} its 50-day moving average")

    price_vs_ma200 = market_features.get("price_vs_ma200")
    if price_vs_ma200 is not None:
        direction = "above" if price_vs_ma200 >= 0 else "below"
        parts.append(f"{abs(price_vs_ma200):.1%} {direction} its 200-day moving average")

    rsi = market_features.get("rsi_14d")
    if rsi is not None:
        if rsi >= 70:
            rsi_note = "overbought"
        elif rsi <= 30:
            rsi_note = "oversold"
        else:
            rsi_note = "neutral"
        parts.append(f"RSI(14) is {rsi:.0f} ({rsi_note})")

    if not parts:
        return "Insufficient technical data for entry context."
    return "As of the latest close, " + "; ".join(parts) + "."


def _technical_stop(current_price: float, market_features: dict, category: str) -> dict:
    """ATR-based stop, falling back to recent swing support when ATR is unavailable."""
    atr = market_features.get("atr_14d")
    support = market_features.get("support_60d")

    if atr:
        multiple = ATR_STOP_MULTIPLE.get(category, 2.5)
        level = current_price - multiple * atr
        method = f"{multiple}x ATR(14) below current price"
        # A nearby swing low that's tighter than the ATR stop (i.e. higher,
        # closer to price) is used instead when available -- it's a level the
        # market has already demonstrated as support, not just a volatility band.
        if support is not None and current_price > support > level:
            return {"level": round(support, 2), "type": "technical", "method": "recent 60-day swing low (tighter than ATR stop)"}
        return {"level": round(level, 2), "type": "technical", "method": method}

    if support is not None and support < current_price:
        return {"level": round(support, 2), "type": "technical", "method": "recent 60-day swing low (ATR unavailable)"}

    return {"level": None, "type": "technical", "method": None, "note": "Insufficient price history (no ATR or swing low) for a technical stop."}


def _technical_target(current_price: float, stop_level: Optional[float], market_features: dict) -> dict:
    resistance = market_features.get("resistance_60d")
    if resistance is not None and resistance > current_price:
        return {"level": round(resistance, 2), "type": "technical", "method": "recent 60-day swing high (resistance)"}

    if stop_level is not None:
        risk = current_price - stop_level
        if risk > 0:
            level = current_price + DEFAULT_REWARD_RISK_MULTIPLE * risk
            return {"level": round(level, 2), "type": "technical", "method": f"{DEFAULT_REWARD_RISK_MULTIPLE}x risk/reward target"}

    return {"level": None, "type": "none", "method": None, "note": "Insufficient data for a technical target."}


def _thesis_invalidation(fundamentals: Optional[dict]) -> dict:
    """Long-horizon 'stop' -- a qualitative fundamental condition, not a price.

    A tight percentage-below-price stop is not meaningful for a multi-year
    thesis (the brief is explicit about this). Instead of fabricating a price
    level from volatility, this states the conditions that would break the
    thesis, referencing today's actual fundamental values where available so
    it isn't a generic template.
    """
    fundamentals = fundamentals or {}

    conditions = []
    revenue_growth = fundamentals.get("revenue_growth")
    if revenue_growth is not None:
        conditions.append(f"revenue growth (currently {revenue_growth:+.1%} YoY) turns sustainedly negative")
    else:
        conditions.append("revenue growth turns sustainedly negative")

    net_margin = fundamentals.get("net_margin")
    if net_margin is not None:
        conditions.append(f"net margin (currently {net_margin:.1%}) compresses structurally, not cyclically")
    else:
        conditions.append("net margin compresses structurally")

    conditions.append("management guidance or competitive position deteriorates in a way that breaks the original growth/margin narrative")

    return {
        "level": None,
        "type": "thesis_invalidation",
        "method": "qualitative fundamental deterioration, not a price level",
        "note": "Exit conditions (any one is a reason to re-underwrite the thesis, not a price trigger): " + "; ".join(conditions) + ".",
    }


def _valuation_exit_note(fundamentals: Optional[dict]) -> str:
    fundamentals = fundamentals or {}
    pe_forward = fundamentals.get("forward_pe")
    peg = fundamentals.get("peg_ratio")
    hint = ""
    if pe_forward is not None:
        hint = f" (forward P/E currently {pe_forward:.1f}"
        hint += f", PEG {peg:.2f})" if peg is not None else ")"
    return (
        "No fixed take-profit -- thesis/valuation based exit. Exit is warranted if the growth/margin/FCF "
        "thesis deteriorates, or if valuation becomes excessive relative to growth" + hint + ", not at a "
        "predetermined price target."
    )


def compute_risk_reward(
    current_price: Optional[float],
    market_features: dict,
    fundamentals: Optional[dict],
    horizon_days: int,
) -> dict:
    """Risk/reward framework for one security at one horizon.

    Returns the fields listed in the brief's section 12, plus explicit
    `type`/`method`/`note` on stop_loss and take_profit so a technical stop is
    never confused with a thesis-invalidation level.
    """
    horizon_days = horizon_mod.clamp_horizon_days(horizon_days)
    category = horizon_mod.horizon_category(horizon_days)
    market_features = market_features or {}

    if current_price is None:
        return {
            "current_price": None,
            "entry_context": _entry_context(None, market_features),
            "stop_loss": {"level": None, "type": None, "method": None, "note": "No current price available."},
            "take_profit": {"level": None, "type": None, "method": None, "note": "No current price available."},
            "risk_per_share": None, "reward_per_share": None, "risk_reward_ratio": None,
            "expected_holding_horizon": _format_horizon(horizon_days),
            "volatility": market_features.get("volatility_1m"),
        }

    if category == "long_term":
        stop_loss = _thesis_invalidation(fundamentals)
        take_profit = {"level": None, "type": "none", "method": "valuation/thesis based", "note": _valuation_exit_note(fundamentals)}
    else:
        stop_loss = _technical_stop(current_price, market_features, category)
        take_profit = _technical_target(current_price, stop_loss.get("level"), market_features)

    risk_per_share = (current_price - stop_loss["level"]) if stop_loss.get("level") is not None else None
    reward_per_share = (take_profit["level"] - current_price) if take_profit.get("level") is not None else None
    risk_reward_ratio = (
        round(reward_per_share / risk_per_share, 2)
        if (risk_per_share and reward_per_share and risk_per_share > 0)
        else None
    )

    return {
        "current_price": current_price,
        "entry_context": _entry_context(current_price, market_features),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_per_share": round(risk_per_share, 2) if risk_per_share is not None else None,
        "reward_per_share": round(reward_per_share, 2) if reward_per_share is not None else None,
        "risk_reward_ratio": risk_reward_ratio,
        "expected_holding_horizon": _format_horizon(horizon_days),
        "volatility": market_features.get("volatility_1m"),
    }
