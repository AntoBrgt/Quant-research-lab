"""PHASE 1 / STEP 14a: small-cap universe, survivorship sensitivity, portfolio math. Synthetic, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import data, model, smallcap as sc


def test_survivorship_sensitivity_math():
    r = pd.Series([0.02, 0.02], index=pd.to_datetime(["2020-01-31", "2020-02-28"]))
    u = pd.Series([0.10, 0.0], index=r.index)
    half = sc.survivorship_adjusted(r, u, 0.50)
    monthly_loss = 0.5 ** (21 / 63) - 1  # -50% over 63 days ~ -20.6% per 21-day month
    assert half.iloc[0] == pytest.approx(0.9 * 0.02 + 0.1 * monthly_loss)
    assert half.iloc[1] == pytest.approx(0.02)  # no unpriced members that month: unchanged
    total = sc.survivorship_adjusted(r, u, 1.00)
    assert total.iloc[0] == pytest.approx(0.9 * 0.02 - 0.1)
    assert (sc.survivorship_adjusted(r, u, 1.0) <= half).all() and (half <= r).all()


def test_universe_filters():
    frame = pd.DataFrame({
        "market_cap": [150e6, 500e6, 500e6, 500e6, 3e9],
        "dollar_volume_20d": [5e6, 5e6, 0.5e6, 5e6, 5e6],
        "raw_price": [10, 10, 10, 2.5, 10],
    })
    assert sc.apply_filters(frame).tolist() == [False, True, False, False, False]


def test_dollar_volume_is_split_invariant():
    idx = pd.bdate_range("2020-01-01", periods=40)
    before = pd.DataFrame({"close_split_adj": 10.0, "volume": 1e6}, index=idx)
    after_split = pd.DataFrame({"close_split_adj": 5.0, "volume": 2e6}, index=idx)  # yfinance's 2:1 back-adjustment
    pd.testing.assert_series_equal(sc.dollar_volume_20d(before), sc.dollar_volume_20d(after_split))


def test_raw_close_undoes_later_splits_only():
    idx = pd.bdate_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"close_split_adj": [50.0, 50.0, 50.0, 50.0], "split": [0, 0, 2.0, 0]}, index=idx)
    assert data.raw_close(prices).tolist() == [100.0, 100.0, 50.0, 50.0]


def test_prefilter_band_is_wider_than_the_real_filter():
    books = pd.DataFrame({"rough_mcap": [10e6, 60e6, 1e9, 3.9e9, 10e9]})
    kept = sc.prefilter(books)["rough_mcap"].tolist()
    assert kept == [60e6, 1e9, 3.9e9]


def test_monthly_portfolio_costs_on_turnover_and_weight_cap():
    rows = []
    for d, rets in (("2020-01-31", [0.10, 0.0, -0.10]), ("2020-02-28", [0.0, 0.0, 0.0])):
        for i, r in enumerate(rets):
            rows.append({"date": pd.Timestamp(d), "ticker": f"T{i}", "score": 3 - i, "ret": r})
    panel = pd.DataFrame(rows)
    port = sc.monthly_portfolio(panel, "score", panel["ret"], top_quantile=0.34, max_weight=0.05)
    first = port.iloc[0]
    assert first["names"] == 1 and first["gross"] == pytest.approx(0.05 * 0.10)  # capped at 5%, rest uninvested
    assert first["net"] == pytest.approx(first["gross"] - 0.05 * sc.COST_PER_SIDE)
    second = port.iloc[1]
    assert second["turnover"] == pytest.approx(0.0, abs=1e-9)  # same name, weight reset to cap: ~no trading


def test_ownership_breadth_change_uses_earlier_months_only():
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]),
                          "ticker": "A", "holders": [3, 4, 5, 7]})
    assert sc.ownership_breadth_change(frame, months=3).tolist()[3] == 4
    assert np.isnan(sc.ownership_breadth_change(frame, months=3).iloc[2])


def test_small_cap_extra_features_pass_the_whitelist_checks():
    model.check_features(model.FEATURES + ["sue", "days_since_filing", "breadth_change_3m"])
    with pytest.raises(ValueError):
        model.check_features(["sue", "fwd_excess"])


def test_embargo_override_widens_but_never_narrows_the_gap():
    dates = pd.bdate_range("2014-01-01", periods=60, freq="21B")
    for embargo, expected_gap in ((4, 5), (1, 4), (None, 4)):  # overlap for 63/21 is 2 -> minimum embargo 3
        for train, test in model.model_splits(dates, 63, 21, min_train=36, test_size=6, embargo=embargo):
            assert dates.get_loc(min(test)) - dates.get_loc(max(train)) == expected_gap
