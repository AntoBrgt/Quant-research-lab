"""PHASE 1b research protocol: holdout guard, append-only registry, DSR, PBO, one-shot final test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import protocol


@pytest.fixture(autouse=True)
def isolated_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "REGISTRY_PATH", tmp_path / "registry.parquet")
    monkeypatch.setattr(protocol, "RETURNS_DIR", tmp_path / "returns")
    monkeypatch.setattr(protocol, "PROMOTIONS_PATH", tmp_path / "promotions.parquet")
    monkeypatch.setattr(protocol, "HOLDOUT_LOG_PATH", tmp_path / "holdout_log.parquet")
    return tmp_path


def _monthly(values, start="2016-01-31"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="ME"))


# ---------------------------------------------------------------- holdout guard


def test_dev_loaders_refuse_holdout_dates(monkeypatch):
    prices = pd.DataFrame({"adj_close": 1.0}, index=pd.bdate_range("2022-12-01", "2024-03-01"))
    monkeypatch.setattr(protocol.data, "load_ext_prices", lambda t: prices)
    dev = protocol.load_prices("AAA")
    assert dev.index.max() < protocol.HOLDOUT_START  # default: stops before the holdout
    with pytest.raises(protocol.HoldoutViolation):
        protocol.load_prices("AAA", end="2024-02-01")
    with pytest.raises(protocol.HoldoutViolation):
        protocol.assert_dev(pd.to_datetime(["2023-06-30", "2024-01-02"]), "feature date")
    protocol.assert_dev(pd.to_datetime(["2023-12-29"]))  # the last dev day is fine


def test_a_label_window_ending_in_the_holdout_is_refused():
    with pytest.raises(protocol.HoldoutViolation):
        protocol.assert_labels_dev(pd.to_datetime(["2023-12-15", "2024-01-16"]))


def test_monthly_panel_is_cut_before_the_holdout(tmp_path):
    frame = pd.DataFrame({"date": pd.to_datetime(["2023-11-30", "2023-12-29", "2024-01-31"]), "x": [1, 2, 3]})
    frame.to_parquet(tmp_path / "m.parquet")
    assert protocol.monthly_members(tmp_path / "m.parquet")["date"].max() < protocol.HOLDOUT_START


# ---------------------------------------------------------------- registry


def test_registry_is_append_only_and_counts_reruns():
    r = _monthly(np.random.default_rng(0).normal(0.01, 0.04, 48))
    a = protocol.register("lgbm", "first", {"lr": 0.03}, r)
    b = protocol.register("lgbm", "same params, re-run", {"lr": 0.03}, r)
    c = protocol.register("lgbm", "other params", {"lr": 0.05}, r)
    reg = protocol.registry()
    assert len(reg) == 3 and len({a, b, c}) == 3 and a.rsplit(":", 1)[0] == b.rsplit(":", 1)[0]
    assert set(protocol.returns_matrix().columns) == {a, b, c}


def test_registering_holdout_returns_is_refused():
    with pytest.raises(protocol.HoldoutViolation):
        protocol.register("x", "leaky", {}, _monthly([0.01] * 30, start="2022-01-31"))


# ---------------------------------------------------------------- DSR and PBO


def test_deflated_sharpe_falls_as_trials_increase():
    rng = np.random.default_rng(1)
    good = _monthly(rng.normal(0.012, 0.04, 96))
    trials = rng.normal(0.0, 0.1, 200)  # per-period Sharpes of unskilled variants
    one = protocol.deflated_sharpe(good, trials[:1], n_trials=1)["dsr"]
    many = protocol.deflated_sharpe(good, trials, n_trials=200)["dsr"]
    assert one > many and 0 <= many <= 1


def test_pbo_high_for_noise_low_for_one_real_edge():
    rng = np.random.default_rng(2)
    noise = pd.DataFrame(rng.normal(0, 0.04, (96, 20)), index=pd.date_range("2016-01-31", periods=96, freq="ME"))
    pbo_noise = protocol.pbo_cscv(noise)["pbo"]
    edge = noise.copy()
    edge[0] = edge[0] + 0.03  # one variant with a strong, persistent edge
    pbo_edge = protocol.pbo_cscv(edge)["pbo"]
    assert pbo_noise > 0.3 and pbo_edge < 0.05


# ---------------------------------------------------------------- final test


def test_final_test_requires_promotion_and_runs_once():
    rng = np.random.default_rng(3)
    noise_ids = [protocol.register("noise", f"n{i}", {"i": i}, _monthly(rng.normal(0, 0.04, 96))) for i in range(5)]
    with pytest.raises(protocol.HoldoutViolation):
        protocol.final_test(noise_ids[0], lambda: {"summary": {}})  # never promoted

    strong = protocol.register("strong", "planted edge", {}, _monthly(rng.normal(0.03, 0.02, 96)))
    decision = protocol.promote(strong)
    assert decision["promoted"], decision

    seen = {}

    def runner():
        seen["open_inside"] = protocol.holdout_is_open()
        protocol.assert_dev(pd.to_datetime(["2025-01-31"]))  # allowed only inside final_test
        return {"summary": {"ok": True}}

    protocol.final_test(strong, runner)
    assert seen["open_inside"] and not protocol.holdout_is_open()
    assert (protocol.holdout_log()["strategy_id"] == strong).sum() == 1
    with pytest.raises(protocol.HoldoutViolation):
        protocol.final_test(strong, runner)  # a second look is refused
