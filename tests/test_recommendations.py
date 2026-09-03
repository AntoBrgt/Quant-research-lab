"""Deterministic recommendation logic against synthetic research -- no LLM."""

import recommendations
import strategy


def _research(signal_count, research_score, risks=None, catalysts=None, features=None):
    return {
        "ticker": "AAPL",
        "signal_count": signal_count,
        "research_score": research_score,
        "risks": risks or [],
        "catalysts": catalysts or [],
        "signals_by_type": {},
        "market_features": features or {},
        "data_freshness": "2026-01-01",
    }


def test_strong_positive_signals_lean_buy():
    research = _research(signal_count=10, research_score=0.8, catalysts=["Strong revenue growth reported."])
    fit = {"score": 0.7, "breakdown": {}, "horizon": "medium_term"}
    rec = recommendations.generate_recommendation("AAPL", research, fit, risk_profile="moderate")
    assert rec.action == "BUY"
    assert rec.asset_signal == 0.8
    assert rec.strategy_fit == 0.7


def test_strong_negative_signals_lean_sell():
    research = _research(signal_count=10, research_score=-0.8, risks=["Severe regulatory investigation risk."])
    fit = {"score": -0.6, "breakdown": {}, "horizon": "medium_term"}
    rec = recommendations.generate_recommendation("AAPL", research, fit, risk_profile="moderate")
    assert rec.action == "SELL"


def test_sparse_signals_return_insufficient_evidence():
    research = _research(signal_count=1, research_score=0.9)  # below MIN_SIGNALS_FOR_ACTION
    fit = {"score": 0.9, "breakdown": {}, "horizon": "medium_term"}
    rec = recommendations.generate_recommendation("AAPL", research, fit, risk_profile="moderate")
    assert rec.action == "INSUFFICIENT_EVIDENCE"
    assert rec.confidence == 0.0


def test_no_signals_at_all_returns_insufficient_evidence():
    research = _research(signal_count=0, research_score=None)
    fit = {"score": None, "breakdown": {}, "horizon": "medium_term"}
    rec = recommendations.generate_recommendation("AAPL", research, fit, risk_profile="moderate")
    assert rec.action == "INSUFFICIENT_EVIDENCE"


def test_four_concepts_stay_separate_fields():
    research = _research(signal_count=10, research_score=0.5, catalysts=["c1"], risks=["r1"])
    fit = {"score": -0.2, "breakdown": {}, "horizon": "medium_term"}
    portfolio_context = {"holdings": {"AAPL": 0.30}}  # large existing position
    rec = recommendations.generate_recommendation("AAPL", research, fit, portfolio_context, risk_profile="moderate")

    # asset_signal, strategy_fit, and portfolio_fit must be independently readable,
    # not collapsed into one number -- and here they disagree (mixed evidence),
    # which is exactly the case that would be lost by a single combined score.
    assert rec.asset_signal == 0.5
    assert rec.strategy_fit == -0.2
    assert rec.portfolio_fit is not None and rec.portfolio_fit < 0  # concentration penalty
    assert rec.asset_signal != rec.strategy_fit != rec.portfolio_fit


def test_conservative_risk_profile_requires_stronger_signal_to_buy():
    research = _research(signal_count=10, research_score=0.35, catalysts=["modest positive signal"])
    fit = {"score": 0.35, "breakdown": {}, "horizon": "medium_term"}

    moderate_rec = recommendations.generate_recommendation("AAPL", research, fit, risk_profile="moderate")
    conservative_rec = recommendations.generate_recommendation("AAPL", research, fit, risk_profile="conservative")

    assert moderate_rec.action == "BUY"
    assert conservative_rec.action != "BUY"  # same evidence, higher bar


def test_new_position_not_currently_held_has_no_penalty():
    research = _research(signal_count=10, research_score=0.5, catalysts=["c1"])
    fit = {"score": 0.5, "breakdown": {}, "horizon": "medium_term"}
    portfolio_context = {"holdings": {"MSFT": 0.20}}  # AAPL not held
    rec = recommendations.generate_recommendation("AAPL", research, fit, portfolio_context, risk_profile="moderate")
    assert rec.portfolio_fit is not None and rec.portfolio_fit > 0


def test_strategy_fit_scoring_is_deterministic():
    research = {
        "signals_by_type": {"revenue_growth": {"count": 2, "avg_strength": 0.8, "score": 0.8}},
        "market_features": {},
    }
    result_1 = strategy.score_strategy_fit(research, "long_term")
    result_2 = strategy.score_strategy_fit(research, "long_term")
    assert result_1 == result_2
