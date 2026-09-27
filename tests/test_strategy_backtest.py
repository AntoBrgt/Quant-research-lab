"""STEP 13b: triple-barrier labels, trade simulation, meta-labeling discipline. Synthetic, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import market_features as mf
import risk_reward
from backtest import barrier_model, barriers, strategy

CAL = pd.bdate_range("2021-01-04", periods=12)


def _ohlc(rows) -> pd.DataFrame:
    """rows: list of (open, high, low, close) on consecutive calendar days."""
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=CAL[: len(rows)])
    return frame.reindex(CAL)


def _walk(rows, stop, target, max_hold=5, signal_pos=0, cost=0.0):
    return barriers.walk_barriers(_ohlc(rows), np.array([signal_pos]), np.array([stop]), np.array([target]), max_hold, cost=cost).iloc[0]


# ---------------------------------------------------------------- barriers


def test_entry_is_the_next_days_open_not_the_signal_close():
    rows = [(100, 101, 99, 100), (103, 104, 102, 103), (103, 104, 102, 103), (103, 104, 102, 103)]
    out = _walk(rows, stop=90, target=120, max_hold=2)
    assert out["entry_price"] == 103 and out["entry_pos"] == 1
    assert out["outcome"] == "time" and out["exit_price"] == 103 and out["days_held"] == 2


def test_same_bar_stop_and_target_counts_as_stop():
    rows = [(100, 100, 100, 100), (100, 100.5, 99.5, 100), (100, 115, 85, 100)]
    out = _walk(rows, stop=90, target=110)
    assert out["outcome"] == "stop" and out["exit_price"] == 90 and out["y"] == 0


def test_target_first_is_y1_and_filled_at_the_target():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100), (101, 111, 100, 110)]
    out = _walk(rows, stop=90, target=110)
    assert out["outcome"] == "target" and out["exit_price"] == 110 and out["y"] == 1 and out["days_held"] == 2


def test_gap_through_the_stop_fills_at_the_worse_open():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100), (80, 82, 79, 81)]
    out = _walk(rows, stop=90, target=110)
    assert out["outcome"] == "stop" and out["exit_price"] == 80


def test_gap_up_through_the_target_gets_no_extra_credit():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100), (125, 126, 124, 125)]
    assert _walk(rows, stop=90, target=110)["exit_price"] == 110


def test_entry_gapping_through_a_barrier_is_not_taken():
    rows = [(100, 100, 100, 100), (89, 90, 88, 89), (100, 100, 100, 100)]
    out = _walk(rows, stop=90, target=110)
    assert out["outcome"] == "skipped" and np.isnan(out["return"])


def test_costs_are_charged_on_entry_and_exit():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100), (100, 101, 99, 100)]
    out = _walk(rows, stop=90, target=110, max_hold=2, cost=barriers.side_cost())
    c = barriers.COST_PER_SIDE + barriers.SLIPPAGE_PER_SIDE
    assert out["return"] == pytest.approx((1 - c) / (1 + c) - 1)  # a flat trade loses ~0.30%
    assert out["return"] == pytest.approx(-0.003, abs=5e-5)
    assert out["r_multiple"] == pytest.approx(out["return"] / 0.10)


def test_delisted_mid_trade_exits_at_last_close():
    rows = [(100, 100, 100, 100), (100, 101, 99, 100), (101, 102, 100, 101)]  # no bars afterwards
    out = _walk(rows, stop=90, target=110, max_hold=5)
    assert out["outcome"] == "data_end" and out["exit_price"] == 101


# ---------------------------------------------------------------- levels: point in time, same as the app


def _prices(n=150, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"adj_close": close, "open": close * (1 + rng.normal(0, 0.005, n)),
                         "high": close * 1.02, "low": close * 0.98, "volume": 1e6}, index=idx)


def test_vectorized_levels_equal_the_apps_as_of_functions():
    prices = _prices()
    levels = barriers.level_inputs(prices)
    for as_of in prices.index[[14, 59, 60, 100, 149]]:
        atr = mf.compute_atr(prices, as_of, 14)
        swing = mf.compute_swing_levels(prices, as_of, 60)
        row = levels.loc[as_of]
        assert (np.isnan(row["atr_14d"]) and atr is None) or row["atr_14d"] == pytest.approx(atr)
        assert (np.isnan(row["support_60d"]) and swing["support"] is None) or row["support_60d"] == pytest.approx(swing["support"])
        assert (np.isnan(row["resistance_60d"]) and swing["resistance"] is None) or row["resistance_60d"] == pytest.approx(swing["resistance"])


def test_app_levels_are_risk_reward_output():
    stop, target = barriers.app_levels(100.0, 2.0, 97.0, 108.0, 14)
    rr = risk_reward.compute_risk_reward(100.0, {"atr_14d": 2.0, "support_60d": 97.0, "resistance_60d": 108.0}, None, 14)
    assert (stop, target) == (rr["stop_loss"]["level"], rr["take_profit"]["level"])


def _panel_for(prices_by_ticker, dates):
    rows = []
    for t in prices_by_ticker:
        for d in dates:
            rows.append({"date": d, "ticker": t, "score_rank": 0.5, "momentum_12_1": 0.1, "return_20d": 0.02})
    return pd.DataFrame(rows)


def test_no_candidate_feature_uses_data_after_the_signal_date():
    prices = _prices(200)
    cal = prices.index
    dates = list(cal[80:190:5])
    shocked = prices.copy()
    cut = cal[120]
    shocked.loc[shocked.index > cut, ["adj_close", "open", "high", "low"]] *= 3
    full, _ = strategy.build_candidates(_panel_for({"AAA": 0}, dates), {"AAA": prices}, cal)
    alt, _ = strategy.build_candidates(_panel_for({"AAA": 0}, dates), {"AAA": shocked}, cal)
    cols = ["close", "atr_14d", "support_60d", "resistance_60d", "app_stop", "app_target", "app_rr", "breakout_20d",
            "stop_dist_pct", "target_dist_pct", "atr_pct"]
    before = full["date"] <= cut
    pd.testing.assert_frame_equal(full.loc[before, cols], alt.loc[before, cols])


# ---------------------------------------------------------------- simulation


def _toy_universe(n_tickers=12, n=260, seed=0):
    prices = {f"T{i:02d}": _prices(n, seed + i) for i in range(n_tickers)}
    cal = next(iter(prices.values())).index
    dates = list(cal[70:n - 25:5])
    rng = np.random.default_rng(seed)
    panel = _panel_for(prices, dates)
    panel["score_rank"] = rng.uniform(-0.5, 0.8, len(panel))
    panel["momentum_12_1"] = rng.normal(size=len(panel))
    return panel, prices, cal


def test_random_baseline_is_reproducible_by_seed():
    panel, prices, cal = _toy_universe()
    cands, ohlc = strategy.build_candidates(panel, prices, cal)
    res = strategy.outcomes(cands, ohlc, strategy.Geometry("grid", 10, 2.0, 2.0))
    closes = strategy.forward_filled_closes(ohlc)
    a = strategy.random_picks(cands, seed=7)
    assert a == strategy.random_picks(cands, seed=7) and a != strategy.random_picks(cands, seed=8)
    s1 = strategy.simulate(cands, res, a, closes, cal)
    s2 = strategy.simulate(cands, res, strategy.random_picks(cands, seed=7), closes, cal)
    pd.testing.assert_series_equal(s1.equity, s2.equity)


def test_simulation_respects_position_limit_sizing_and_cash():
    panel, prices, cal = _toy_universe()
    cands, ohlc = strategy.build_candidates(panel, prices, cal)
    res = strategy.outcomes(cands, ohlc, strategy.Geometry("grid", 20, 1.5, 3.0))
    closes = strategy.forward_filled_closes(ohlc)
    sim = strategy.simulate(cands, res, strategy.weekly_picks(cands, strategy.pick_momentum, k=5), closes, cal, max_positions=3)
    trades = sim.trades
    assert len(trades) > 0 and sim.not_opened > 0
    # Never more than 3 positions open at once.
    events = sorted([(d, 1) for d in trades["entry_date"]] + [(d, -1) for d in trades["exit_date"]], key=lambda e: (e[0], -e[1]))
    open_count, peak = 0, 0
    for _, delta in events:
        open_count += delta
        peak = max(peak, open_count)
    assert peak <= 3
    assert (trades["weight"] <= strategy.MAX_POSITION_WEIGHT + 1e-9).all()
    assert (sim.equity > 0).all()


def test_bootstrap_ci_and_percentile():
    trades = pd.DataFrame({"signal_date": np.repeat(pd.bdate_range("2021-01-01", periods=50, freq="5B"), 3),
                           "r_multiple": np.random.default_rng(0).normal(0.3, 1.0, 150)})
    lo, hi = strategy.bootstrap_expectancy_ci(trades)
    assert lo < trades["r_multiple"].mean() < hi and lo > 0
    assert strategy.percentile_of(5.0, range(10)) == 50.0


# ---------------------------------------------------------------- meta-labeling


def _ml_setup():
    panel, prices, cal = _toy_universe(n_tickers=30, n=900, seed=3)
    cands, ohlc = strategy.build_candidates(panel, prices, cal)
    res = strategy.outcomes(cands, ohlc, strategy.Geometry("app", 10))
    return cands, res


def test_embargo_training_trades_exit_before_the_first_test_signal(monkeypatch):
    cands, res = _ml_setup()
    seen = []
    real_fit = barrier_model._fit

    def spy(x, y, num_trees):
        seen.append(x.index)
        return real_fit(x, y, 5)

    monkeypatch.setattr(barrier_model, "_fit", spy)
    # Inner threshold fits are covered by the scrambled-label test; here every _fit is a fold's final model.
    monkeypatch.setattr(barrier_model, "_choose_threshold", lambda *a, **k: 0.0)
    probs = barrier_model.walk_forward_probabilities(cands, res, 10, 5, min_train=40, test_size=20, num_trees=5)
    folds = list(probs.dropna(subset=["fold"]).groupby("fold"))
    assert len(folds) > 1 and len(seen) == len(folds)
    for (fold, rows), train_idx in zip(folds, seen):
        first_test = cands.loc[rows.index, "signal_pos"].min()
        assert (res.loc[train_idx, "exit_pos"] < first_test).all()  # no training trade still open at the first test signal
        assert cands.loc[train_idx, "date"].max() < cands.loc[rows.index, "date"].min()
    assert barrier_model.embargo_dates(10, 5) == 3 and barrier_model.embargo_dates(20, 5) == 5


def test_test_fold_outcomes_never_influence_predictions_or_thresholds():
    cands, res = _ml_setup()
    a = barrier_model.walk_forward_probabilities(cands, res, 10, 5, min_train=40, test_size=20, num_trees=10)
    first_test_date = cands.loc[a["fold"] == 0, "date"].min()
    scrambled = res.copy()
    later = cands["date"] >= first_test_date
    scrambled.loc[later & (res["outcome"] != "skipped"), "y"] = 1 - res.loc[later & (res["outcome"] != "skipped"), "y"]
    b = barrier_model.walk_forward_probabilities(cands, scrambled, 10, 5, min_train=40, test_size=20, num_trees=10)
    fold0 = a["fold"] == 0
    pd.testing.assert_series_equal(a.loc[fold0, "p_target"], b.loc[fold0, "p_target"])
    assert a.loc[fold0, "threshold"].iloc[0] == b.loc[fold0, "threshold"].iloc[0]


def test_meta_features_use_the_whitelist_and_no_labels():
    cands, _ = _ml_setup()
    x = barrier_model.design_matrix(cands.assign(fwd_excess=1.0, y=1, r_multiple=2.0))
    assert not {"fwd_excess", "y", "r_multiple", "return", "outcome"} & set(x.columns)
    assert list(x.columns) == barrier_model.model.FEATURES + barrier_model.GEOMETRY_FEATURES


def test_auc_and_calibration():
    y = np.array([0, 0, 1, 1])
    assert barrier_model.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert barrier_model.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    rng = np.random.default_rng(0)
    p = pd.Series(rng.uniform(size=1000))
    table = barrier_model.calibration_table(pd.Series((rng.uniform(size=1000) < p).astype(int)), p)
    assert len(table) == 10 and table["actual_hit_rate"].iloc[-1] > table["actual_hit_rate"].iloc[0]
