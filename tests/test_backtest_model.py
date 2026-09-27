"""STEP 12 learned model -- out-of-sample discipline first, fit quality second. Synthetic, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import evaluate, model

LABEL_DAYS, REBALANCE_EVERY = 63, 21


def _synthetic_pit_panel(signal_strength: float, dates=60, names=120, seed=0, shuffle_labels=False) -> pd.DataFrame:
    """A PIT-shaped panel where only `momentum_12_1` (optionally) predicts fwd_excess."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2019-01-01", periods=dates, freq="21B"):
        signal = rng.normal(size=names)
        excess = signal_strength * signal + rng.normal(size=names)
        for i in range(names):
            row = {f: rng.normal() for f in model.FEATURES}
            row.update({"date": d, "ticker": f"T{i:03d}", "momentum_12_1": signal[i], "fwd_excess": excess[i],
                        "fwd_return": excess[i] + 0.01, "score_rank": rng.normal(), "score_full": rng.normal()})
            rows.append(row)
    panel = pd.DataFrame(rows)
    if shuffle_labels:
        panel["fwd_excess"] = rng.permutation(panel["fwd_excess"].to_numpy())
    return panel


def _oos_summary(panel: pd.DataFrame) -> dict:
    predictions, _ = model.walk_forward_predict(panel, LABEL_DAYS, REBALANCE_EVERY, num_trees=100)
    rows = panel.merge(predictions, on=["date", "ticker"])
    return evaluate.summarize(evaluate.per_date_metrics(rows, "score_ml"), evaluate.overlap_lags(LABEL_DAYS, REBALANCE_EVERY))


def test_planted_signal_is_learned_out_of_sample():
    summary = _oos_summary(_synthetic_pit_panel(0.5))
    assert summary["mean_rank_ic"] > 0.3 and summary["rank_ic_tstat_nw"] > 5


def test_pure_noise_is_not_learned():
    summary = _oos_summary(_synthetic_pit_panel(0.0, seed=1))
    assert abs(summary["mean_rank_ic"]) < 0.05


def test_shuffled_labels_give_no_oos_ic():
    summary = _oos_summary(_synthetic_pit_panel(0.5, seed=2, shuffle_labels=True))
    assert abs(summary["mean_rank_ic"]) < 0.05


def test_no_test_date_is_in_its_training_set_and_the_embargo_holds():
    dates = pd.bdate_range("2019-01-01", periods=60, freq="21B")
    embargo = evaluate.overlap_lags(LABEL_DAYS, REBALANCE_EVERY) + 1
    folds = list(model.model_splits(dates, LABEL_DAYS, REBALANCE_EVERY))
    assert folds
    for train, test in folds:
        assert not set(train) & set(test)
        assert max(train) < min(test)
        gap = dates.get_loc(min(test)) - dates.get_loc(max(train))
        assert gap == embargo + 1
        # The last training label (entered the next day, held LABEL_DAYS) ends before the first test date.
        assert (gap * REBALANCE_EVERY) > LABEL_DAYS + 1


def test_rows_outside_test_folds_have_no_prediction():
    panel = _synthetic_pit_panel(0.5, dates=50)
    predictions, _ = model.walk_forward_predict(panel, LABEL_DAYS, REBALANCE_EVERY, num_trees=20)
    first_test = min(test[0] for _, test in model.model_splits(panel["date"].unique(), LABEL_DAYS, REBALANCE_EVERY))
    assert predictions.loc[predictions["date"] < first_test, ["score_ml", "score_linear"]].isna().all().all()
    assert predictions.loc[predictions["date"] >= first_test, "score_ml"].notna().all()


@pytest.mark.parametrize("bad", ["fwd_excess", "fwd_return", "label", "date", "ticker", "score_rank", "score_ml", "fwd_anything"])
def test_forbidden_columns_can_never_be_features(bad):
    with pytest.raises(ValueError):
        model.check_features(model.FEATURES + [bad])
    with pytest.raises(ValueError):
        model.feature_matrix(_synthetic_pit_panel(0.0, dates=2, names=5), [bad])


def test_feature_matrix_is_exactly_the_whitelist_even_if_labels_are_present(monkeypatch):
    panel = _synthetic_pit_panel(0.0, dates=3, names=10)
    x = model.feature_matrix(panel)
    assert list(x.columns) == model.FEATURES
    monkeypatch.setattr(model, "FEATURES", model.FEATURES + ["fwd_excess"])
    with pytest.raises(ValueError):
        model.walk_forward_predict(panel, LABEL_DAYS, REBALANCE_EVERY)


def test_cross_sectional_ranks_are_per_date_in_unit_interval_and_keep_nan():
    frame = pd.DataFrame({
        "date": ["d1"] * 4 + ["d2"] * 2,
        "x": [10.0, 30.0, np.nan, 20.0, 5.0, 1.0],
    })
    ranked = model.cross_sectional_rank(frame, ["x"])["x"]
    assert ranked.tolist()[:2] == [0.0, 1.0] and np.isnan(ranked[2]) and ranked[3] == 0.5
    assert ranked.tolist()[4:] == [1.0, 0.0]


def test_model_refuses_the_current_universe_panel():
    panel = _synthetic_pit_panel(0.0, dates=3, names=10).drop(columns=["score_rank", "institutional_direction_score"])
    with pytest.raises(ValueError, match="pit"):
        model.walk_forward_predict(panel, LABEL_DAYS, REBALANCE_EVERY)
