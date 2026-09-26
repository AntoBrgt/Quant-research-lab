"""Fundamental/technical/institutional labels stay independent, visible components
(section 17: "do not hide the individual components behind one score")."""

import pandas as pd

import research_engine


def _extended(market_features=None, fundamentals=None, institutional_rows=None, horizon_score=0.3):
    return {
        "research": {"market_features": market_features or {}},
        "fundamentals": fundamentals or {},
        "institutional_context": pd.DataFrame(institutional_rows) if institutional_rows is not None else pd.DataFrame(),
        "horizon_fit": {"score": horizon_score},
    }


def test_all_components_insufficient_evidence_with_no_data():
    views = research_engine.component_views(_extended())
    assert views["fundamental"] == "INSUFFICIENT_EVIDENCE"
    assert views["technical"] == "INSUFFICIENT_EVIDENCE"
    assert views["institutional"] == "INSUFFICIENT_EVIDENCE"


def test_technical_positive_from_uptrend_features():
    extended = _extended(market_features={"price_vs_ma50": 0.10, "price_vs_ma200": 0.20, "return_60d": 0.08})
    views = research_engine.component_views(extended)
    assert views["technical"] == "POSITIVE"


def test_fundamental_negative_from_deteriorating_factors():
    extended = _extended(fundamentals={"revenue_growth": -0.15, "net_margin": -0.05})
    views = research_engine.component_views(extended)
    assert views["fundamental"] == "NEGATIVE"


def test_institutional_label_reuses_universe_aggregation():
    extended = _extended(institutional_rows=[{"view_direction": "POSITIVE"}, {"view_direction": "POSITIVE"}])
    views = research_engine.component_views(extended)
    assert views["institutional"] == "POSITIVE"


def test_components_can_disagree_independently():
    extended = _extended(
        market_features={"price_vs_ma50": 0.10, "price_vs_ma200": 0.10, "return_60d": 0.10},  # positive technical
        fundamentals={"revenue_growth": -0.2, "net_margin": -0.1},  # negative fundamental
    )
    views = research_engine.component_views(extended)
    assert views["technical"] == "POSITIVE"
    assert views["fundamental"] == "NEGATIVE"
    assert views["technical"] != views["fundamental"]  # disagreement is preserved, not averaged away


def test_horizon_fit_score_passed_through_unmodified():
    extended = _extended(horizon_score=0.42)
    views = research_engine.component_views(extended)
    assert views["horizon_fit_score"] == 0.42


def test_confidence_is_insufficient_evidence_with_no_components():
    result = research_engine.research_confidence({"fundamental": "INSUFFICIENT_EVIDENCE", "technical": "INSUFFICIENT_EVIDENCE", "institutional": "INSUFFICIENT_EVIDENCE"}, signal_count=0)
    assert result["level"] == "Insufficient evidence"


def test_confidence_is_high_when_components_agree_with_enough_signals():
    result = research_engine.research_confidence({"fundamental": "POSITIVE", "technical": "POSITIVE", "institutional": "POSITIVE"}, signal_count=10)
    assert result["level"] == "High"


def test_confidence_is_medium_when_components_disagree():
    result = research_engine.research_confidence({"fundamental": "POSITIVE", "technical": "NEGATIVE", "institutional": "NEUTRAL"}, signal_count=10)
    assert result["level"] == "Medium"
    assert any("disagree" in d for d in result["drivers"])


def test_confidence_is_low_with_only_one_component():
    result = research_engine.research_confidence({"fundamental": "POSITIVE", "technical": "INSUFFICIENT_EVIDENCE", "institutional": "INSUFFICIENT_EVIDENCE"}, signal_count=2)
    assert result["level"] == "Low"


def test_confidence_is_never_expressed_as_a_probability():
    result = research_engine.research_confidence({"fundamental": "POSITIVE", "technical": "POSITIVE", "institutional": "POSITIVE"}, signal_count=10)
    assert result["level"] in ("High", "Medium", "Low", "Insufficient evidence")
    assert not isinstance(result["level"], float)
