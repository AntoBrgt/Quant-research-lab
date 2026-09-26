"""Numeric investment horizon (1 day -> 20 years) and horizon-aware scoring.

`strategy.py` (short_term/medium_term/long_term) stays as-is -- it's what the
existing portfolio flow and its tests already depend on. This module is the
new, continuous generalization the institutional-universe/company-research
flow uses: a horizon is a number of days, not one of three buckets, and the
weighting an analysis places on signals/technicals/fundamentals shifts
smoothly as that number changes, per the brief's "the same stock should
potentially produce different research conclusions depending on horizon."

Design: a handful of named anchor points, each an explicit, independently
readable weight profile (not one hardcoded formula). A horizon between two
anchors gets a log-time-weighted blend of their profiles -- log-space because
the difference between a 1-day and 30-day horizon matters far more than the
difference between a 9-year and 10-year one. Every weight is a plain module-
level dict, meant to be read and tuned, not a black box.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# The numeric horizon parameter
# ---------------------------------------------------------------------------

# Predefined points a UI slider/selectbox can offer -- convenience only. The
# underlying parameter is always a plain integer number of days; any value
# in [MIN_HORIZON_DAYS, MAX_HORIZON_DAYS] is valid, not just these presets.
HORIZON_PRESETS: dict[str, int] = {
    "1 day": 1,
    "3 days": 3,
    "1 week": 7,
    "2 weeks": 14,
    "1 month": 30,
    "3 months": 91,
    "6 months": 182,
    "1 year": 365,
    "2 years": 730,
    "5 years": 1825,
    "10 years": 3650,
    "20 years": 7300,
}

MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 7300


def clamp_horizon_days(horizon_days: int) -> int:
    return max(MIN_HORIZON_DAYS, min(MAX_HORIZON_DAYS, int(horizon_days)))


def horizon_category(horizon_days: int) -> str:
    """Coarse bucket, used only for display copy and risk/reward methodology
    selection (section 12/13 of the brief) -- never for scoring itself, which
    uses the continuous interpolation below.
    """
    horizon_days = clamp_horizon_days(horizon_days)
    if horizon_days <= 5:
        return "very_short_term"
    if horizon_days <= 90:
        return "short_term"
    if horizon_days <= 730:
        return "medium_term"
    return "long_term"


# ---------------------------------------------------------------------------
# Anchor weight profiles
# ---------------------------------------------------------------------------

# Each anchor is a full, independently-tunable profile. Weights need not sum
# to 1 -- scoring normalizes by total weight actually available, same
# convention as strategy.py.
_ANCHOR_PROFILES: dict[int, dict] = {
    1: {  # very short: price action / momentum / event risk dominate
        "signal_types": {"guidance": 0.3, "risk": 0.5, "management_confidence": 0.1},
        "market_features": {"return_1d": 1.0, "return_5d": 0.6, "volume_ratio": 0.7, "volatility_1m": -0.4},
        "fundamentals": {},
    },
    30: {  # short/medium: earnings momentum, guidance, near-term catalysts
        "signal_types": {"guidance": 1.0, "demand": 0.8, "management_confidence": 0.6, "risk": 0.6},
        "market_features": {"return_1m": 1.0, "return_20d": 0.6, "volume_ratio": 0.4, "volatility_1m": -0.2},
        "fundamentals": {"forward_pe": -0.15},
    },
    365: {  # ~1y: earnings, revenue growth, guidance, valuation, trend
        "signal_types": {"earnings": 1.0, "revenue_growth": 1.0, "guidance": 0.8, "margins": 0.7, "competition": 0.4},
        "market_features": {"return_3m": 0.5, "return_6m": 0.5, "price_vs_ma50": 0.3},
        "fundamentals": {
            "revenue_growth": 0.6,
            "net_margin": 0.3,
            "forward_pe": -0.3,
        },
    },
    3650: {  # 10y+ (also covers up to 20y): structural, FCF, balance sheet, moat
        "signal_types": {
            "revenue_growth": 1.0, "margins": 0.8, "cash_flow": 0.8, "debt": 0.6,
            "competition": 0.6, "management_confidence": 0.5, "regulation": 0.4,
        },
        "market_features": {"return_1y": 0.3, "price_vs_ma200": 0.2},
        "fundamentals": {
            "revenue_growth": 0.8,
            "fcf_growth": 0.7,
            "roe": 0.6,
            "fcf_margin": 0.6,
            "net_debt_to_market_cap": -0.4,
            "fcf_yield": 0.5,
        },
    },
}

_ANCHOR_DAYS_SORTED = sorted(_ANCHOR_PROFILES)


def _log_interp_weight(horizon_days: int, lo_days: int, hi_days: int) -> float:
    """Fraction of the way from lo to hi, in log space. 0 at lo, 1 at hi."""
    if lo_days == hi_days:
        return 0.0
    log_lo, log_hi, log_h = math.log(lo_days), math.log(hi_days), math.log(horizon_days)
    return max(0.0, min(1.0, (log_h - log_lo) / (log_hi - log_lo)))


def weight_profile_for_horizon(horizon_days: int) -> dict:
    """Blend the two nearest anchor profiles for an arbitrary horizon_days.

    Below the shortest anchor or above the longest, the nearest anchor's
    profile is used as-is (no extrapolation past the defined range).
    """
    horizon_days = clamp_horizon_days(horizon_days)

    if horizon_days <= _ANCHOR_DAYS_SORTED[0]:
        return _ANCHOR_PROFILES[_ANCHOR_DAYS_SORTED[0]]
    if horizon_days >= _ANCHOR_DAYS_SORTED[-1]:
        return _ANCHOR_PROFILES[_ANCHOR_DAYS_SORTED[-1]]

    lo_days = max(d for d in _ANCHOR_DAYS_SORTED if d <= horizon_days)
    hi_days = min(d for d in _ANCHOR_DAYS_SORTED if d >= horizon_days)
    if lo_days == hi_days:
        return _ANCHOR_PROFILES[lo_days]

    blend = _log_interp_weight(horizon_days, lo_days, hi_days)
    lo_profile, hi_profile = _ANCHOR_PROFILES[lo_days], _ANCHOR_PROFILES[hi_days]

    blended: dict[str, dict] = {}
    for group in ("signal_types", "market_features", "fundamentals"):
        merged: dict[str, float] = {}
        keys = set(lo_profile.get(group, {})) | set(hi_profile.get(group, {}))
        for key in keys:
            lo_w = lo_profile.get(group, {}).get(key, 0.0)
            hi_w = hi_profile.get(group, {}).get(key, 0.0)
            merged[key] = lo_w * (1 - blend) + hi_w * blend
        blended[group] = merged
    return blended


# ---------------------------------------------------------------------------
# Feature normalization (raw value -> roughly [-1, 1])
# ---------------------------------------------------------------------------

def normalize_market_feature(name: str, value: float) -> float:
    """Same idea as strategy.py's `_normalized_feature`, extended to the
    additional technical features. Thresholds are hand-picked, documented
    heuristics -- not statistically fit -- and are meant to be tuned.
    """
    if name.startswith("relative_return_"):
        # A relative (vs-benchmark) return is usually smaller in magnitude
        # than the equivalent absolute return, so it's scored on the
        # matching absolute-return horizon's scale, not a separate one.
        name = name.removeprefix("relative_")
    if name in ("return_1d", "return_5d", "return_20d", "return_60d", "return_1m", "return_3m", "return_6m"):
        scale = {"return_1d": 20, "return_5d": 10, "return_20d": 6, "return_60d": 4}.get(name, 5)
        return max(-1.0, min(1.0, value * scale))
    if name in ("return_1y", "return_252d"):
        return max(-1.0, min(1.0, value * 2))  # +/-50% annual move maps to +/-1
    if name in ("volatility_1m", "volatility_20d", "volatility_60d"):
        return max(-1.0, min(1.0, (value - 0.3) * -2))  # above ~30% annualized reads as a headwind
    if name == "volume_ratio":
        return max(-1.0, min(1.0, value - 1.0))
    if name in ("price_vs_ma50", "price_vs_ma200"):
        return max(-1.0, min(1.0, value * 5))
    return 0.0


def normalize_fundamental(path: str, value: float) -> float:
    """Heuristic normalization for fundamental ratios. Documented, tunable
    defaults -- see module docstring. `None` inputs are filtered out by the
    caller before this is reached.
    """
    if path in ("revenue_growth", "fcf_growth", "eps_growth", "operating_income_growth"):
        return max(-1.0, min(1.0, value * 5))  # +/-20% growth maps to +/-1
    if path == "net_margin":
        return max(-1.0, min(1.0, value * 4))  # 25% net margin ~ +1
    if path == "roe":
        return max(-1.0, min(1.0, value * 4))  # 25% ROE ~ +1
    if path == "fcf_margin":
        return max(-1.0, min(1.0, value * 5))  # 20% FCF margin ~ +1
    if path == "fcf_yield":
        return max(-1.0, min(1.0, value * 12))  # ~8% FCF yield ~ +1 (cheap/cash-generative)
    if path == "forward_pe":
        if value is None or value <= 0:
            return 0.0
        return max(-1.0, min(1.0, (25 - value) / 25))  # PE 0 -> +1, PE 25 -> 0, PE 50 -> -1
    if path == "net_debt_to_market_cap":
        return max(-1.0, min(1.0, -value * 2))  # net debt = 50% of market cap -> -1
    return 0.0


def get_nested(d: dict, dotted_path: str) -> Optional[float]:
    node = d
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, (int, float)) else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_horizon_fit(research: dict, fundamentals: Optional[dict], horizon_days: int) -> dict:
    """Blend signal, technical, and fundamental evidence for one horizon.

    Returns {"score": float|None, "breakdown": {...}, "horizon_days": int,
    "horizon_category": str}. `breakdown` keeps every component readable --
    this is the numeric-horizon analog of `strategy.score_strategy_fit`, kept
    as a separate function/module rather than merged into `strategy.py` so
    the existing horizon buckets and their tests are untouched.
    """
    horizon_days = clamp_horizon_days(horizon_days)
    weights = weight_profile_for_horizon(horizon_days)

    breakdown: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    signals_by_type = research.get("signals_by_type", {}) if research else {}
    for signal_type, weight in weights.get("signal_types", {}).items():
        if weight == 0 or signal_type not in signals_by_type:
            continue
        signal_score = signals_by_type[signal_type].get("score")
        if signal_score is None:
            continue
        breakdown[f"signal:{signal_type}"] = signal_score
        weighted_sum += signal_score * weight
        total_weight += abs(weight)

    market_features = research.get("market_features", {}) if research else {}
    for feature_name, weight in weights.get("market_features", {}).items():
        if weight == 0:
            continue
        raw_value = market_features.get(feature_name)
        if raw_value is None:
            continue
        normalized = normalize_market_feature(feature_name, raw_value)
        breakdown[f"feature:{feature_name}"] = normalized
        weighted_sum += normalized * weight
        total_weight += abs(weight)

    fundamentals = fundamentals or {}
    for path, weight in weights.get("fundamentals", {}).items():
        if weight == 0:
            continue
        if path == "net_debt_to_market_cap":
            net_debt = fundamentals.get("net_debt")
            market_cap = fundamentals.get("market_cap")
            raw_value = (net_debt / market_cap) if (net_debt is not None and market_cap) else None
        else:
            raw_value = get_nested(fundamentals, path)
        if raw_value is None:
            continue
        normalized = normalize_fundamental(path, raw_value)
        breakdown[f"fundamental:{path}"] = normalized
        weighted_sum += normalized * weight
        total_weight += abs(weight)

    if total_weight == 0:
        return {"score": None, "breakdown": breakdown, "horizon_days": horizon_days, "horizon_category": horizon_category(horizon_days)}

    return {
        "score": round(weighted_sum / total_weight, 4),
        "breakdown": breakdown,
        "horizon_days": horizon_days,
        "horizon_category": horizon_category(horizon_days),
    }


# ---------------------------------------------------------------------------
# Horizon-aware component weighting (group-level, explainable)
# ---------------------------------------------------------------------------
#
# Everything above this point (the log-interpolated per-field anchors) is
# unchanged and keeps serving `research_engine.load_extended_research()`
# exactly as before. This section adds a second, coarser-grained, more
# directly explainable weighting: instead of ~15 individually-weighted raw
# fields blended continuously, it groups technicals/fundamentals into 10
# named, interpretable components (5 technical, 5 fundamental) and weights
# those groups from 5 fixed, fully transparent horizon profiles. This is what
# lets a caller show "why does a 1-day view differ from a 10-year view" as
# a handful of named, inspectable numbers rather than a diff of ~15 raw
# feature weights.

HorizonProfile = Literal["VERY_SHORT", "SHORT", "MEDIUM", "LONG", "VERY_LONG"]

# Upper bound (inclusive, in days) for each profile except the last, which
# catches everything above the highest threshold. Chosen so every named
# preset in HORIZON_PRESETS lands in the profile the brief specifies:
# 1-3 days -> VERY_SHORT, 1-2 weeks -> SHORT, 1-6 months -> MEDIUM,
# 1-2 years -> LONG, 5-20 years -> VERY_LONG.
_PROFILE_DAY_THRESHOLDS: list[tuple[int, HorizonProfile]] = [
    (3, "VERY_SHORT"),
    (14, "SHORT"),
    (182, "MEDIUM"),
    (730, "LONG"),
]


def horizon_profile(horizon_days: int) -> HorizonProfile:
    """Discrete 5-way profile for a horizon -- see module docstring for why
    this is separate from the 4-way `horizon_category()` used by
    `risk_reward.py`'s stop/target methodology selection (left untouched).
    """
    horizon_days = clamp_horizon_days(horizon_days)
    for threshold, profile in _PROFILE_DAY_THRESHOLDS:
        if horizon_days <= threshold:
            return profile
    return "VERY_LONG"


# Which existing technicals.py / fundamentals.py fields feed each group.
# Deliberately a handful of representative fields per group, not every field
# those modules expose -- this stays a small, readable weighting layer, not
# an accumulation of every available metric.
GROUP_FIELDS: dict[str, tuple[str, ...]] = {
    "technical_trend": ("price_vs_ma50", "price_vs_ma200"),
    "technical_momentum": ("return_20d", "return_60d", "return_252d"),
    "technical_volatility": ("volatility_20d", "volatility_60d"),
    "technical_volume": ("volume_ratio",),
    "technical_relative_strength": ("relative_return_20d", "relative_return_60d", "relative_return_252d"),
    "fundamental_growth": ("revenue_growth", "eps_growth"),
    "fundamental_profitability": ("net_margin", "roe"),
    "fundamental_cash_flow": ("fcf_margin", "fcf_growth"),
    "fundamental_balance_sheet": ("net_debt_to_market_cap",),  # synthesized, see _group_field_value
    "fundamental_valuation": ("forward_pe", "fcf_yield"),
}

TECHNICAL_GROUPS = tuple(g for g in GROUP_FIELDS if g.startswith("technical_"))
FUNDAMENTAL_GROUPS = tuple(g for g in GROUP_FIELDS if g.startswith("fundamental_"))

# Explicit, inspectable weights per profile. Each profile's weights sum to
# 1.0 across all 10 groups (see test_horizon.py) -- this is the *nominal*
# weighting; `compute_horizon_weighted_view` renormalizes over whatever
# groups actually have data for a given company (section 6: missing data is
# excluded, never treated as a negative signal or padded to keep the total
# artificially at 1.0).
PROFILE_WEIGHTS: dict[HorizonProfile, dict[str, float]] = {
    "VERY_SHORT": {  # technical analysis dominates -- a 1-3 day view is about current market behavior
        "technical_trend": 0.20, "technical_momentum": 0.25, "technical_volatility": 0.15,
        "technical_volume": 0.15, "technical_relative_strength": 0.10,
        "fundamental_growth": 0.03, "fundamental_profitability": 0.03, "fundamental_cash_flow": 0.03,
        "fundamental_balance_sheet": 0.03, "fundamental_valuation": 0.03,
    },
    "SHORT": {  # still technical-led, but valuation/growth/profitability start to matter
        "technical_trend": 0.18, "technical_momentum": 0.20, "technical_volatility": 0.12,
        "technical_volume": 0.10, "technical_relative_strength": 0.10,
        "fundamental_growth": 0.08, "fundamental_profitability": 0.07, "fundamental_cash_flow": 0.03,
        "fundamental_balance_sheet": 0.02, "fundamental_valuation": 0.10,
    },
    "MEDIUM": {  # substantially balanced
        "technical_trend": 0.13, "technical_momentum": 0.14, "technical_volatility": 0.06,
        "technical_volume": 0.04, "technical_relative_strength": 0.13,
        "fundamental_growth": 0.14, "fundamental_profitability": 0.12, "fundamental_cash_flow": 0.08,
        "fundamental_balance_sheet": 0.04, "fundamental_valuation": 0.12,
    },
    "LONG": {  # fundamentals dominate; technicals kept for entry/trend/momentum/relative strength context
        "technical_trend": 0.08, "technical_momentum": 0.07, "technical_volatility": 0.02,
        "technical_volume": 0.01, "technical_relative_strength": 0.07,
        "fundamental_growth": 0.18, "fundamental_profitability": 0.16, "fundamental_cash_flow": 0.16,
        "fundamental_balance_sheet": 0.10, "fundamental_valuation": 0.15,
    },
    "VERY_LONG": {  # fundamentals overwhelmingly dominate; technicals remain visible as secondary context only
        "technical_trend": 0.03, "technical_momentum": 0.03, "technical_volatility": 0.01,
        "technical_volume": 0.01, "technical_relative_strength": 0.02,
        "fundamental_growth": 0.22, "fundamental_profitability": 0.20, "fundamental_cash_flow": 0.20,
        "fundamental_balance_sheet": 0.12, "fundamental_valuation": 0.16,
    },
}


def profile_weight_totals(profile: HorizonProfile) -> dict[str, float]:
    """Nominal (data-independent) technical vs fundamental weight totals for
    a profile -- pure function of the profile, used to verify the *intended*
    shift in emphasis regardless of what data any particular company has.
    """
    weights = PROFILE_WEIGHTS[profile]
    return {
        "technical": round(sum(weights[g] for g in TECHNICAL_GROUPS), 4),
        "fundamental": round(sum(weights[g] for g in FUNDAMENTAL_GROUPS), 4),
    }


def _group_field_value(group: str, field: str, technicals: dict, fundamentals: dict) -> Optional[float]:
    if group in TECHNICAL_GROUPS:
        return technicals.get(field)
    if field == "net_debt_to_market_cap":
        net_debt, market_cap = fundamentals.get("net_debt"), fundamentals.get("market_cap")
        return (net_debt / market_cap) if (net_debt is not None and market_cap) else None
    return fundamentals.get(field)


def compute_group_scores(technicals: Optional[dict], fundamentals: Optional[dict]) -> dict[str, Optional[float]]:
    """One normalized [-1, 1] score per component group, or `None` if every
    field feeding that group is missing -- reuses the exact same
    `normalize_market_feature`/`normalize_fundamental` heuristics already
    used by `score_horizon_fit`, never a second normalization scheme.
    """
    technicals, fundamentals = technicals or {}, fundamentals or {}
    scores: dict[str, Optional[float]] = {}
    for group, fields in GROUP_FIELDS.items():
        normalized_values = []
        for field in fields:
            raw_value = _group_field_value(group, field, technicals, fundamentals)
            if raw_value is None:
                continue
            normalizer = normalize_market_feature if group in TECHNICAL_GROUPS else normalize_fundamental
            normalized_values.append(normalizer(field, raw_value))
        scores[group] = round(sum(normalized_values) / len(normalized_values), 4) if normalized_values else None
    return scores


def compute_horizon_weighted_view(technicals: Optional[dict], fundamentals: Optional[dict], horizon_days: int) -> dict:
    """The horizon-aware, group-level, fully explainable view.

    Returns horizon/profile identification, the nominal and actually-used
    (post-missing-data-renormalization) technical/fundamental weight totals,
    a `score` (weighted average of available group scores, `None` if nothing
    is available), per-group `components` (score/weight/contribution, so
    nothing is collapsed into one opaque number), and `dominant_factors` --
    the groups with the largest absolute contribution for this horizon.

    Missing data (section 6): a group with no resolvable field is excluded
    from the weighted average and does not lower the score -- the total
    weight actually used renormalizes to whatever groups have real data,
    it does not stay pinned at the nominal profile total.
    """
    horizon_days = clamp_horizon_days(horizon_days)
    profile = horizon_profile(horizon_days)
    weights = PROFILE_WEIGHTS[profile]
    group_scores = compute_group_scores(technicals, fundamentals)

    components: dict[str, dict] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for group, weight in weights.items():
        score = group_scores.get(group)
        if score is None or weight == 0:
            continue
        contribution = round(score * weight, 4)
        components[group] = {"score": score, "weight": weight, "contribution": contribution}
        weighted_sum += contribution
        total_weight += weight

    overall_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else None

    technical_weight = round(sum(c["weight"] for g, c in components.items() if g in TECHNICAL_GROUPS), 4)
    fundamental_weight = round(sum(c["weight"] for g, c in components.items() if g in FUNDAMENTAL_GROUPS), 4)
    technical_contribution = round(sum(c["contribution"] for g, c in components.items() if g in TECHNICAL_GROUPS), 4)
    fundamental_contribution = round(sum(c["contribution"] for g, c in components.items() if g in FUNDAMENTAL_GROUPS), 4)

    dominant_factors = sorted(components, key=lambda g: abs(components[g]["contribution"]), reverse=True)[:3]

    return {
        "horizon_days": horizon_days,
        "horizon_profile": profile,
        "score": overall_score,
        "technical_weight": technical_weight,
        "fundamental_weight": fundamental_weight,
        "technical_contribution": technical_contribution,
        "fundamental_contribution": fundamental_contribution,
        "component_weights": dict(weights),
        "components": components,
        "dominant_factors": dominant_factors,
    }
