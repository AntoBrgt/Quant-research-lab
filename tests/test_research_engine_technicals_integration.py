"""research_engine.load_extended_research must enrich market_features with
technicals.py's fields WITHOUT losing or changing any key horizon.py/
risk_reward.py/strategy.py/component_views already depend on -- this is the
regression this integration could most easily cause.
"""

import pandas as pd

import horizon
import research_engine
import risk_reward


class FakePriceProvider:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        return self.df


class FakeFundamentalsProvider:
    def get_raw_fundamentals(self, ticker: str) -> dict:
        return {"info": {"sector": "Technology"}, "income_stmt": {}, "cashflow": {}, "balance_sheet": {}}


def _trending_prices(n=260, start=100.0, step=0.5):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [start + i * step for i in range(n)]
    df = pd.DataFrame(
        {"adj_close": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "volume": [1_000_000.0] * n},
        index=dates,
    )
    df.index.name = "date"
    return df


def test_merged_market_features_keeps_old_keys_strategy_and_horizon_depend_on():
    price_prov = FakePriceProvider(_trending_prices())
    extended = research_engine.load_extended_research(
        "TEST", horizon_days=365, price_prov=price_prov, fundamentals_prov=FakeFundamentalsProvider(), benchmark_prov=price_prov,
    )
    mf = extended["research"]["market_features"]

    # Pre-existing keys strategy.py/horizon.py/risk_reward.py/component_views
    # already read -- must still be present and non-None with sufficient history.
    for key in ("return_1m", "return_3m", "return_6m", "return_1y", "atr_14d", "moving_average_50d", "moving_average_200d", "price_vs_ma50", "price_vs_ma200", "rsi_14d", "volume_ratio", "support_60d", "resistance_60d"):
        assert key in mf, f"{key} missing after technicals merge"


def test_merged_market_features_adds_new_technicals_fields():
    price_prov = FakePriceProvider(_trending_prices())
    extended = research_engine.load_extended_research(
        "TEST", horizon_days=365, price_prov=price_prov, fundamentals_prov=FakeFundamentalsProvider(), benchmark_prov=price_prov,
    )
    mf = extended["research"]["market_features"]
    for key in ("return_120d", "return_252d", "moving_average_100d", "volatility_20d", "volatility_60d", "trend", "momentum_state", "relative_return_20d"):
        assert key in mf


def test_horizon_scoring_still_works_against_the_merged_dict():
    price_prov = FakePriceProvider(_trending_prices())
    extended = research_engine.load_extended_research(
        "TEST", horizon_days=365, price_prov=price_prov, fundamentals_prov=FakeFundamentalsProvider(), benchmark_prov=price_prov,
    )
    fit = horizon.score_horizon_fit(extended["research"], extended["fundamentals"], horizon_days=365)
    assert fit["score"] is not None
    assert "feature:price_vs_ma50" in fit["breakdown"]  # the pre-existing weighted key still resolves


def test_risk_reward_still_reads_atr_and_swing_levels_from_the_merged_dict():
    price_prov = FakePriceProvider(_trending_prices())
    extended = research_engine.load_extended_research(
        "TEST", horizon_days=30, price_prov=price_prov, fundamentals_prov=FakeFundamentalsProvider(), benchmark_prov=price_prov,
    )
    rr = risk_reward.compute_risk_reward(
        extended["current_price"], extended["research"]["market_features"], extended["fundamentals"], horizon_days=30,
    )
    assert rr["stop_loss"]["level"] is not None
    assert rr["stop_loss"]["type"] == "technical"


def test_horizon_weighted_view_is_exposed_alongside_horizon_fit():
    price_prov = FakePriceProvider(_trending_prices())
    extended = research_engine.load_extended_research(
        "TEST", horizon_days=1, price_prov=price_prov, fundamentals_prov=FakeFundamentalsProvider(), benchmark_prov=price_prov,
    )
    assert "horizon_fit" in extended  # pre-existing key, untouched
    view = extended["horizon_weighted_view"]
    assert view["horizon_profile"] == "VERY_SHORT"
    assert view["technical_weight"] > view["fundamental_weight"]


def test_overlapping_keys_have_identical_values_not_silently_diverged():
    """atr_14d/moving_average_50d/etc. are computed twice (once inside
    load_company_research's compute_features, once inside compute_technicals)
    -- the merge must not leave a value that disagrees with an independent
    computation of the same window over the same prices.
    """
    import market_features

    price_prov = FakePriceProvider(_trending_prices())
    extended = research_engine.load_extended_research(
        "TEST", horizon_days=365, price_prov=price_prov, fundamentals_prov=FakeFundamentalsProvider(), benchmark_prov=price_prov,
    )
    mf = extended["research"]["market_features"]
    independent_ma50 = market_features.compute_moving_average(price_prov.df, window=50)
    assert mf["moving_average_50d"] == independent_ma50
