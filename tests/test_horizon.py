"""Numeric horizon parameter + horizon-aware scoring -- deterministic, no LLM."""

import horizon


def test_horizon_category_boundaries():
    assert horizon.horizon_category(1) == "very_short_term"
    assert horizon.horizon_category(5) == "very_short_term"
    assert horizon.horizon_category(6) == "short_term"
    assert horizon.horizon_category(90) == "short_term"
    assert horizon.horizon_category(91) == "medium_term"
    assert horizon.horizon_category(730) == "medium_term"
    assert horizon.horizon_category(731) == "long_term"
    assert horizon.horizon_category(7300) == "long_term"


def test_clamp_horizon_days_stays_in_range():
    assert horizon.clamp_horizon_days(0) == horizon.MIN_HORIZON_DAYS
    assert horizon.clamp_horizon_days(999_999) == horizon.MAX_HORIZON_DAYS
    assert horizon.clamp_horizon_days(365) == 365


def test_weight_profile_at_an_exact_anchor_matches_that_anchor():
    profile = horizon.weight_profile_for_horizon(1)
    assert profile["signal_types"]["risk"] == 0.5  # from the 1-day anchor, unblended


def test_weight_profile_between_anchors_is_a_blend():
    at_short = horizon.weight_profile_for_horizon(30)["market_features"]["return_1m"]
    at_medium_ish = horizon.weight_profile_for_horizon(180)["market_features"].get("return_1m", 0.0)
    # 180 days sits between the 30-day and 365-day anchors; the 365-day anchor
    # doesn't weight return_1m at all, so the blended weight must be strictly
    # less than the 30-day anchor's own weight, not equal to either endpoint.
    assert 0 <= at_medium_ish < at_short


def test_score_horizon_fit_returns_none_with_no_data():
    result = horizon.score_horizon_fit(research={}, fundamentals={}, horizon_days=30)
    assert result["score"] is None
    assert result["horizon_days"] == 30


def test_score_is_deterministic():
    research = {"signals_by_type": {"guidance": {"score": 0.6}}, "market_features": {}}
    r1 = horizon.score_horizon_fit(research, {}, 30)
    r2 = horizon.score_horizon_fit(research, {}, 30)
    assert r1 == r2


def test_same_evidence_scores_differently_across_horizons():
    """The central claim of section 9: the same company can score differently
    depending on horizon. Here, a strong near-term guidance signal plus a
    weak/negative long-term cash-flow story should read positively short-term
    and negatively long-term.
    """
    research = {
        "signals_by_type": {
            "guidance": {"score": 0.9},   # only weighted at short horizons
            "cash_flow": {"score": -0.8},  # only weighted at long horizons
        },
        "market_features": {},
    }
    short_result = horizon.score_horizon_fit(research, {}, horizon_days=30)
    long_result = horizon.score_horizon_fit(research, {}, horizon_days=3650)

    assert short_result["score"] > 0
    assert long_result["score"] < 0


def test_breakdown_keeps_components_visible_not_one_opaque_score():
    research = {"signals_by_type": {"guidance": {"score": 0.5}}, "market_features": {"return_1m": 0.1}}
    result = horizon.score_horizon_fit(research, {}, horizon_days=30)
    assert "signal:guidance" in result["breakdown"]
    assert "feature:return_1m" in result["breakdown"]


def test_fundamental_factor_contributes_at_long_horizon_only():
    fundamentals = {"revenue_growth": 0.15}
    short_result = horizon.score_horizon_fit({}, fundamentals, horizon_days=1)
    long_result = horizon.score_horizon_fit({}, fundamentals, horizon_days=3650)
    assert "fundamental:revenue_growth" not in short_result["breakdown"]
    assert "fundamental:revenue_growth" in long_result["breakdown"]
    assert long_result["score"] > 0


# ---------------------------------------------------------------------------
# Horizon-aware group weighting (5 profiles) -- Step 10 of the brief
# ---------------------------------------------------------------------------

def test_horizon_preset_to_profile_mapping():
    expected = {
        "1 day": "VERY_SHORT", "3 days": "VERY_SHORT",
        "1 week": "SHORT", "2 weeks": "SHORT",
        "1 month": "MEDIUM", "3 months": "MEDIUM", "6 months": "MEDIUM",
        "1 year": "LONG", "2 years": "LONG",
        "5 years": "VERY_LONG", "10 years": "VERY_LONG", "20 years": "VERY_LONG",
    }
    for label, expected_profile in expected.items():
        horizon_days = horizon.HORIZON_PRESETS[label]
        assert horizon.horizon_profile(horizon_days) == expected_profile, f"{label} ({horizon_days}d)"


def test_every_profile_weight_set_sums_to_one():
    for profile, weights in horizon.PROFILE_WEIGHTS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, profile


def test_every_profile_covers_all_ten_groups():
    for profile, weights in horizon.PROFILE_WEIGHTS.items():
        assert set(weights) == set(horizon.GROUP_FIELDS), profile


def test_very_short_is_technical_dominated():
    totals = horizon.profile_weight_totals("VERY_SHORT")
    assert totals["technical"] > totals["fundamental"]


def test_very_long_is_fundamental_dominated():
    totals = horizon.profile_weight_totals("VERY_LONG")
    assert totals["fundamental"] > totals["technical"]


def test_technical_weight_decreases_monotonically_as_horizon_lengthens():
    profiles_in_order = ["VERY_SHORT", "SHORT", "MEDIUM", "LONG", "VERY_LONG"]
    technical_weights = [horizon.profile_weight_totals(p)["technical"] for p in profiles_in_order]
    assert technical_weights == sorted(technical_weights, reverse=True)
    assert technical_weights[0] > technical_weights[-1]  # strictly, not just non-increasing


def test_fundamental_weight_increases_monotonically_as_horizon_lengthens():
    profiles_in_order = ["VERY_SHORT", "SHORT", "MEDIUM", "LONG", "VERY_LONG"]
    fundamental_weights = [horizon.profile_weight_totals(p)["fundamental"] for p in profiles_in_order]
    assert fundamental_weights == sorted(fundamental_weights)
    assert fundamental_weights[0] < fundamental_weights[-1]


def test_medium_is_substantially_balanced():
    totals = horizon.profile_weight_totals("MEDIUM")
    assert abs(totals["technical"] - totals["fundamental"]) <= 0.1


# --- missing data (section 6) -----------------------------------------------

def test_missing_group_is_excluded_not_treated_as_negative():
    """A company with zero technicals available (e.g. JPM-style missing ROIC
    situation, generalized to a whole side missing) must not be penalized --
    the score should reflect only the fundamentals that DO exist.
    """
    fundamentals = {"revenue_growth": 0.10, "net_margin": 0.20}
    result_with_technicals = horizon.compute_horizon_weighted_view({"price_vs_ma50": 0.05}, fundamentals, horizon_days=365)
    result_without_technicals = horizon.compute_horizon_weighted_view(None, fundamentals, horizon_days=365)

    assert result_without_technicals["score"] is not None
    assert result_without_technicals["technical_weight"] == 0.0
    assert "technical_trend" not in result_without_technicals["components"]
    assert "fundamental_growth" in result_without_technicals["components"]


def test_fully_missing_data_returns_none_score_not_zero():
    result = horizon.compute_horizon_weighted_view(None, None, horizon_days=30)
    assert result["score"] is None
    assert result["components"] == {}


def test_none_field_is_excluded_from_its_group_not_averaged_as_zero():
    # revenue_growth present, eps_growth missing -- fundamental_growth's score
    # must reflect revenue_growth alone, not (revenue_growth + 0) / 2.
    scores_full = horizon.compute_group_scores({}, {"revenue_growth": 0.20, "eps_growth": 0.20})
    scores_partial = horizon.compute_group_scores({}, {"revenue_growth": 0.20, "eps_growth": None})
    assert scores_full["fundamental_growth"] == scores_partial["fundamental_growth"]


# --- explainability -----------------------------------------------------------

def test_output_exposes_required_explainability_fields():
    result = horizon.compute_horizon_weighted_view({"price_vs_ma50": 0.05}, {"revenue_growth": 0.1}, horizon_days=30)
    for key in ("horizon_days", "horizon_profile", "technical_weight", "fundamental_weight", "technical_contribution", "fundamental_contribution", "component_weights", "components", "dominant_factors"):
        assert key in result


def test_no_opaque_single_score_is_the_only_output():
    result = horizon.compute_horizon_weighted_view({"price_vs_ma50": 0.05, "return_20d": 0.03}, {"revenue_growth": 0.1}, horizon_days=30)
    assert len(result["components"]) >= 2  # more than one visible component behind the overall score
    for group, detail in result["components"].items():
        assert set(detail) == {"score", "weight", "contribution"}


def test_dominant_factors_reflects_largest_contributions():
    # Strong technical_momentum vs a much smaller technical_trend tilt -- both
    # groups are present, momentum's contribution must rank first.
    technicals = {"return_20d": 0.15, "return_60d": 0.15, "return_252d": 0.15, "price_vs_ma50": 0.01}
    result = horizon.compute_horizon_weighted_view(technicals, {}, horizon_days=1)
    assert "technical_trend" in result["components"]
    assert result["dominant_factors"][0] == "technical_momentum"


def test_result_is_deterministic():
    technicals, fundamentals = {"price_vs_ma50": 0.05, "return_20d": 0.02}, {"revenue_growth": 0.1, "net_margin": 0.2}
    r1 = horizon.compute_horizon_weighted_view(technicals, fundamentals, horizon_days=180)
    r2 = horizon.compute_horizon_weighted_view(technicals, fundamentals, horizon_days=180)
    assert r1 == r2


def test_same_company_same_data_scores_differently_by_horizon():
    """The point of the whole task: identical technicals+fundamentals produce
    a different weighted view depending on horizon.
    """
    technicals = {"return_20d": 0.20, "return_60d": 0.20, "return_252d": 0.20, "price_vs_ma50": 0.10, "price_vs_ma200": 0.10}
    fundamentals = {"revenue_growth": -0.10, "eps_growth": -0.10, "net_margin": -0.05}  # weak fundamentals
    very_short = horizon.compute_horizon_weighted_view(technicals, fundamentals, horizon_days=1)
    very_long = horizon.compute_horizon_weighted_view(technicals, fundamentals, horizon_days=3650)

    assert very_short["score"] > 0  # strong near-term technicals dominate
    assert very_long["score"] < 0  # weak fundamentals dominate at a 10-year horizon
    assert very_short["technical_weight"] > very_long["technical_weight"]
    assert very_short["fundamental_weight"] < very_long["fundamental_weight"]
