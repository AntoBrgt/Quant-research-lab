"""Strategy scoring: how well a company's research fits a chosen time horizon.

Weights are one configurable dict (`STRATEGY_WEIGHTS`), not scattered constants
across the codebase, so tuning a horizon's emphasis doesn't require touching
logic. Scoring itself is deterministic Python over `research_engine.py`'s
output -- no LLM call here, which is also what keeps it unit-testable.
"""

from __future__ import annotations

from typing import Literal

Horizon = Literal["short_term", "medium_term", "long_term"]

# Each horizon weights signal types (from research["signals_by_type"]) and a
# couple of market features (from research["market_features"]). Weights need
# not sum to 1 -- score_strategy_fit normalizes by total weight actually used.
STRATEGY_WEIGHTS: dict[str, dict] = {
    "short_term": {
        "signal_types": {
            "guidance": 1.0,
            "demand": 0.8,
            "management_confidence": 0.6,
            "risk": 0.6,
        },
        "market_features": {
            "return_1m": 1.0,
            "volume_ratio": 0.6,
            "volatility_1m": -0.4,  # high recent volatility is a headwind, not a tailwind
        },
    },
    "medium_term": {
        "signal_types": {
            "earnings": 1.0,
            "revenue_growth": 1.0,
            "guidance": 0.8,
            "margins": 0.7,
            "competition": 0.4,
        },
        "market_features": {
            "return_3m": 0.6,
            "rsi_14d": 0.0,  # informative for risk, not scored directly here
        },
    },
    "long_term": {
        "signal_types": {
            "revenue_growth": 1.0,
            "margins": 0.8,
            "cash_flow": 0.8,
            "debt": 0.6,
            "competition": 0.6,
            "management_confidence": 0.5,
            "regulation": 0.4,
        },
        "market_features": {},
    },
}


def _normalized_feature(name: str, value: float) -> float:
    """Map a raw feature onto roughly [-1, 1] so it's comparable to signal scores."""
    if value is None:
        return 0.0
    if name == "return_1m" or name == "return_3m":
        return max(-1.0, min(1.0, value * 5))  # +/-20% move maps to +/-1
    if name == "volatility_1m":
        return max(-1.0, min(1.0, (value - 0.3) * -2))  # above ~30% annualized reads negative
    if name == "volume_ratio":
        return max(-1.0, min(1.0, (value - 1.0)))
    return 0.0


def score_strategy_fit(research: dict, horizon: Horizon) -> dict:
    """Return {"score": float in [-1, 1] or None, "breakdown": {...}}."""
    if horizon not in STRATEGY_WEIGHTS:
        raise ValueError(f"Unknown horizon: {horizon}. Use one of {list(STRATEGY_WEIGHTS)}.")

    weights = STRATEGY_WEIGHTS[horizon]
    breakdown: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    signals_by_type = research.get("signals_by_type", {})
    for signal_type, weight in weights["signal_types"].items():
        if signal_type not in signals_by_type or weight == 0:
            continue
        signal_score = signals_by_type[signal_type].get("score")
        if signal_score is None:
            continue
        breakdown[f"signal:{signal_type}"] = signal_score
        weighted_sum += signal_score * weight
        total_weight += abs(weight)

    market_features = research.get("market_features", {})
    for feature_name, weight in weights["market_features"].items():
        if weight == 0:
            continue
        raw_value = market_features.get(feature_name)
        if raw_value is None:
            continue
        normalized = _normalized_feature(feature_name, raw_value)
        breakdown[f"feature:{feature_name}"] = normalized
        weighted_sum += normalized * weight
        total_weight += abs(weight)

    if total_weight == 0:
        return {"score": None, "breakdown": breakdown, "horizon": horizon}

    return {"score": round(weighted_sum / total_weight, 4), "breakdown": breakdown, "horizon": horizon}
