"""PHASE 1b addendum: Holm correction, decile spread after costs, verdicts, one-shot holdout gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import phase1b_addendum as add
from backtest import protocol


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "ADDENDUM_LOG_PATH", tmp_path / "addendum.parquet")


def test_holm_matches_hand_computation():
    adj = add.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotone: never below the previous step


def test_a_single_t_of_2_is_the_threshold():
    assert add.two_sided_p(2.0) == pytest.approx(add.HOLM_ALPHA)


def _stats(t=None, ic=0.02, spread=0.01, **over):
    base = {v: {"mean_rank_ic": ic, "ic_t": t if t is not None else 0.5, "spread_mean": spread} for v in add.VARIANTS}
    for v, s in over.items():
        base[v] = {**base[v], **s}
    return base


def test_t_of_2_5_fails_after_holm_over_11_but_t_of_4_passes():
    v = "lgbm_reg:7bb578be22:0"
    weak = add.verdicts(_stats(**{v: {"ic_t": 2.5}}))
    assert not weak[v]["checks"]["holm_t_above_2"]  # p 0.012 x 11 > 0.0455
    strong = add.verdicts(_stats(**{v: {"ic_t": 4.0}}))
    assert strong[v]["passed"]


def test_negative_t_never_passes_and_spread_must_be_positive():
    v = "ridge:6190916c75:0"
    assert not add.verdicts(_stats(**{v: {"ic_t": -5.0, "mean_rank_ic": -0.05}}))[v]["passed"]
    assert not add.verdicts(_stats(**{v: {"ic_t": 5.0, "spread_mean": -0.001}}))[v]["passed"]


def test_lazy_prices_needs_a_text_pass_that_beats_the_same_model_without_text():
    text, without = "lgbm_reg:7bb578be22:0", "lgbm_reg:4e59bfdeef:0"
    stats = _stats(**{text: {"ic_t": 5.0, "mean_rank_ic": 0.03}, without: {"ic_t": 5.0, "mean_rank_ic": 0.04}})
    v = add.verdicts(stats)
    assert v[text]["passed"]
    assert not add.lazy_prices_verdict(stats, v)["passed"]  # text passes but adds nothing
    stats[without]["mean_rank_ic"] = 0.01
    assert add.lazy_prices_verdict(stats, add.verdicts(stats))["passed"]


def test_decile_spread_charges_costs_on_both_legs():
    dates = pd.to_datetime(["2024-01-31"] * 20 + ["2024-02-29"] * 20)
    score = np.tile(np.arange(20, dtype=float), 2)
    rows = pd.DataFrame({"date": dates, "ticker": [f"T{i}" for i in range(20)] * 2, "s": score,
                         "next_return": np.where(score >= 18, 0.05, np.where(score <= 1, -0.05, 0.0))})
    spread = add.decile_spread(rows, "s", cost=0.001)
    # month 1: 10% spread minus a full buy of each leg (1.0 turnover x 0.1% each); month 2: same names, equal drift -> no trade
    assert spread.iloc[0] == pytest.approx(0.10 - 0.001 - 0.001)
    assert spread.iloc[1] == pytest.approx(0.10)


def test_addendum_holdout_opens_once_and_logs_before_running():
    seen = {}

    def runner():
        seen["open"] = protocol.holdout_is_open()
        seen["logged"] = protocol.ADDENDUM_LOG_PATH.exists()
        return {"summary": {"x": 1}}

    result = protocol.addendum_holdout("t1", runner)
    assert seen == {"open": True, "logged": True} and result["looks"] == 1
    assert not protocol.holdout_is_open()
    with pytest.raises(protocol.HoldoutViolation):
        protocol.addendum_holdout("t1", runner)
    assert protocol.addendum_holdout("t1", runner, force=True)["looks"] == 2


def test_a_crashed_run_still_counts_as_a_look():
    def boom():
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        protocol.addendum_holdout("t2", boom)
    with pytest.raises(protocol.HoldoutViolation):
        protocol.addendum_holdout("t2", lambda: {"summary": {}})


def test_variants_are_exactly_the_registered_eleven():
    assert len(add.VARIANTS) == 11
    reg = protocol.data.BACKTEST_DIR / "registry.parquet"
    if reg.exists():
        assert set(pd.read_parquet(reg)["variant_id"]) == set(add.VARIANTS)
