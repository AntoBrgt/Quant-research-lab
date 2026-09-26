"""Risk/reward framework: horizon determines methodology, never a fixed % stop."""

import risk_reward


def test_short_horizon_uses_atr_based_technical_stop():
    market_features = {"atr_14d": 2.0, "resistance_60d": 120.0, "support_60d": 90.0}
    result = risk_reward.compute_risk_reward(current_price=100.0, market_features=market_features, fundamentals={}, horizon_days=5)

    assert result["stop_loss"]["type"] == "technical"
    assert result["stop_loss"]["level"] < 100.0
    assert result["take_profit"]["type"] == "technical"
    assert result["take_profit"]["level"] > 100.0
    assert result["risk_reward_ratio"] is not None
    assert result["risk_per_share"] > 0
    assert result["reward_per_share"] > 0


def test_short_horizon_falls_back_to_swing_low_without_atr():
    market_features = {"support_60d": 92.0}  # no atr_14d
    result = risk_reward.compute_risk_reward(current_price=100.0, market_features=market_features, fundamentals={}, horizon_days=5)
    assert result["stop_loss"]["level"] == 92.0
    assert "ATR unavailable" in result["stop_loss"]["method"]


def test_short_horizon_with_no_technical_data_reports_insufficient_data_honestly():
    result = risk_reward.compute_risk_reward(current_price=100.0, market_features={}, fundamentals={}, horizon_days=5)
    assert result["stop_loss"]["level"] is None
    assert "Insufficient" in result["stop_loss"]["note"]
    assert result["risk_reward_ratio"] is None


def test_long_horizon_uses_thesis_invalidation_not_a_technical_stop():
    fundamentals = {"revenue_growth": 0.12, "net_margin": 0.22}
    market_features = {"atr_14d": 2.0, "support_60d": 90.0}  # present but must be ignored at this horizon
    result = risk_reward.compute_risk_reward(current_price=100.0, market_features=market_features, fundamentals=fundamentals, horizon_days=3650)

    assert result["stop_loss"]["type"] == "thesis_invalidation"
    assert result["stop_loss"]["level"] is None  # never a fabricated price level
    assert "revenue growth" in result["stop_loss"]["note"]
    assert "12.0%" in result["stop_loss"]["note"]  # references the actual current value, not a template


def test_long_horizon_has_no_fixed_take_profit():
    result = risk_reward.compute_risk_reward(current_price=100.0, market_features={}, fundamentals={}, horizon_days=3650)
    assert result["take_profit"]["level"] is None
    assert result["take_profit"]["type"] == "none"
    assert "No fixed take-profit" in result["take_profit"]["note"]
    assert result["risk_reward_ratio"] is None


def test_no_current_price_is_handled_without_crashing():
    result = risk_reward.compute_risk_reward(current_price=None, market_features={}, fundamentals={}, horizon_days=30)
    assert result["current_price"] is None
    assert result["stop_loss"]["level"] is None
    assert result["risk_reward_ratio"] is None


def test_expected_holding_horizon_is_human_readable():
    assert risk_reward.compute_risk_reward(100.0, {}, {}, 1)["expected_holding_horizon"] == "1 day"
    assert risk_reward.compute_risk_reward(100.0, {}, {}, 365)["expected_holding_horizon"] == "~1 year"
    assert "year" in risk_reward.compute_risk_reward(100.0, {}, {}, 3650)["expected_holding_horizon"]
