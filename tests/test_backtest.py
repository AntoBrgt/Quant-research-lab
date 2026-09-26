"""STEP 11 backtest: point-in-time correctness first, metrics second."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import horizon
from backtest import dataset, evaluate, labels, pit_features


def _random_walk(n=800, seed=0, start="2018-01-01", drift=0.0003, vol=0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    closes = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    df = pd.DataFrame(
        {"adj_close": closes, "high": closes * 1.01, "low": closes * 0.99, "open": closes,
         "volume": rng.integers(800_000, 1_200_000, n).astype(float)},
        index=dates,
    )
    df.index.name = "date"
    return df


def _raw_fundamentals() -> dict:
    periods = ["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31"]
    return {
        "info": {"marketCap": 1e11},
        "income_stmt": {
            "Total Revenue": dict(zip(periods, [100.0, 110.0, 121.0, 150.0])),
            "Net Income": dict(zip(periods, [10.0, 11.0, 12.0, 20.0])),
            "Diluted EPS": dict(zip(periods, [1.0, 1.1, 1.2, 2.0])),
        },
        "cashflow": {"Free Cash Flow": dict(zip(periods, [8.0, 9.0, 10.0, 15.0]))},
        "balance_sheet": {
            "Stockholders Equity": dict(zip(periods, [50.0, 55.0, 60.0, 70.0])),
            "Cash And Cash Equivalents": dict(zip(periods, [5.0, 6.0, 7.0, 8.0])),
            "Total Debt": dict(zip(periods, [20.0, 20.0, 20.0, 20.0])),
        },
    }


# --- point-in-time features ---------------------------------------------------

def test_pit_technicals_ignore_everything_after_as_of():
    prices = _random_walk()
    as_of = prices.index[500]
    shocked = prices.copy()
    shocked.loc[shocked.index > as_of, ["adj_close", "high", "low"]] *= 5  # a future crash/rally
    bench = _random_walk(seed=1)
    assert pit_features.pit_technicals(prices, as_of, bench) == pit_features.pit_technicals(shocked, as_of, bench)


def test_pit_technicals_empty_for_stale_or_delisted_ticker():
    prices = _random_walk(n=400)
    assert pit_features.pit_technicals(prices, prices.index[-1] + pd.Timedelta(days=30)) == {}


def test_momentum_12_1_skips_the_last_month():
    prices = _random_walk(n=300)
    as_of = prices.index[-1]
    closes = prices["adj_close"]
    expected = closes.iloc[-22] / closes.iloc[-253] - 1
    assert pit_features.pit_technicals(prices, as_of)["momentum_12_1"] == pytest.approx(expected)


def test_fundamentals_are_invisible_until_the_reporting_lag_has_passed():
    raw = _raw_fundamentals()
    # FY2022 ends 2022-12-31; with a 90-day lag it becomes usable on 2023-03-31.
    before = pit_features.pit_fundamentals(raw, "2023-03-01", lag_days=90)
    after = pit_features.pit_fundamentals(raw, "2023-04-15", lag_days=90)
    assert before["revenue_growth"] == pytest.approx(0.10)  # FY2021 vs FY2020
    assert after["revenue_growth"] == pytest.approx(150 / 121 - 1)  # FY2022 now public
    assert before["fundamentals_period_end"] == "2021-12-31"


def test_fundamentals_before_any_filing_is_public_are_all_missing():
    out = pit_features.pit_fundamentals(_raw_fundamentals(), "2020-01-15")
    assert all(out[k] is None for k in ("revenue_growth", "net_margin", "roe", "fcf_margin"))


def test_forward_pe_is_never_reconstructed():
    assert pit_features.pit_fundamentals(_raw_fundamentals(), "2024-01-01")["forward_pe"] is None


def test_historical_market_cap_scales_with_price_ratio():
    out = pit_features.pit_fundamentals(_raw_fundamentals(), "2024-01-01", price_at_as_of=50.0, price_now=100.0)
    assert out["fcf_yield"] == pytest.approx(15.0 / 5e10)


# --- labels -------------------------------------------------------------------

def test_forward_return_enters_on_the_next_trading_day():
    dates = pd.bdate_range("2024-01-01", periods=10)
    stock = pd.Series([100, 200, 110, 120, 130, 140, 150, 160, 170, 180], index=dates, dtype=float)
    bench = pd.Series([100, 100, 100, 100, 100, 105, 105, 105, 105, 105], index=dates, dtype=float)
    ret, excess = labels.forward_returns(stock, bench, dates[0], days=3)
    # entry = dates[1] (200), exit = dates[4] (130): the signal-day close is never the entry.
    assert ret == pytest.approx(130 / 200 - 1)
    assert excess == pytest.approx(130 / 200 - 1 - 0.0)


def test_forward_return_is_missing_when_the_window_runs_past_the_data():
    dates = pd.bdate_range("2024-01-01", periods=5)
    s = pd.Series(range(1, 6), index=dates, dtype=float)
    assert labels.forward_returns(s, s, dates[2], days=5) == (None, None)


def test_delisted_stock_gets_no_label_after_the_fill_limit():
    calendar = pd.bdate_range("2024-01-01", periods=40)
    prices = pd.DataFrame({"adj_close": np.linspace(10, 20, 10)}, index=calendar[:10])
    aligned = labels.aligned_closes(prices, calendar)
    assert aligned.iloc[9 + labels.MAX_FILL_DAYS] == pytest.approx(20)
    assert pd.isna(aligned.iloc[10 + labels.MAX_FILL_DAYS])


# --- panel --------------------------------------------------------------------

def _panel(tickers=("AAA", "BBB", "CCC"), n=800, cut=None):
    series = {t: _random_walk(n=n, seed=i + 10) for i, t in enumerate(tickers)}
    bench = _random_walk(n=n, seed=99)
    if cut is not None:
        series = {t: p.loc[:cut] for t, p in series.items()}
        bench = bench.loc[:cut]

    def fundamentals_loader(ticker):
        # "Today's" market cap always matches "today's" last price, as it does live.
        raw = _raw_fundamentals()
        raw["info"] = {"marketCap": 1e9 * float(series[ticker]["adj_close"].iloc[-1])}
        return raw

    return dataset.build_panel(
        tickers, price_loader=series.__getitem__, fundamentals_loader=fundamentals_loader,
        benchmark=bench, rebalance_every=21, label_days=63,
    )


def test_panel_features_do_not_change_when_future_data_is_removed():
    full = _panel(n=1300)
    dates = full["date"].sort_values().unique()
    cut = dates[40]  # late enough that point-in-time fundamentals exist
    assert full.loc[full["date"] <= cut, "has_fundamentals"].any()
    truncated = _panel(n=1300, cut=cut)
    feature_cols = [c for c in truncated.columns if c not in ("fwd_return", "fwd_excess")]
    a = full[full["date"] <= cut][feature_cols].reset_index(drop=True)
    b = truncated[feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_panel_score_is_the_production_horizon_score():
    panel = _panel()
    row = panel.dropna(subset=["score_technical"]).iloc[0]
    tech = {f: row[f] for fs in horizon.GROUP_FIELDS.values() for f in fs if f in row and pd.notna(row[f])}
    expected = horizon.compute_horizon_weighted_view(tech, None, dataset.DEFAULT_HORIZON_DAYS)["score"]
    assert row["score_technical"] == pytest.approx(expected)


def test_panel_skips_a_ticker_whose_prices_fail_to_load():
    series = {"AAA": _random_walk(n=600)}
    panel = dataset.build_panel(["AAA", "BAD"], price_loader=series.__getitem__, fundamentals_loader=None,
                                benchmark=_random_walk(n=600, seed=5))
    assert set(panel["ticker"]) == {"AAA"}


# --- evaluation -----------------------------------------------------------------

def _synthetic_eval_panel(signal_strength: float, dates=24, names=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2020-01-01", periods=dates, freq="21B"):
        signal = rng.normal(size=names)
        excess = signal_strength * signal + rng.normal(size=names)
        rows += [{"date": d, "ticker": f"T{i}", "sig": s, "fwd_excess": e} for i, (s, e) in enumerate(zip(signal, excess))]
    return pd.DataFrame(rows)


def test_planted_signal_is_detected_and_noise_is_not():
    strong = evaluate.summarize(evaluate.per_date_metrics(_synthetic_eval_panel(0.5), "sig"), overlap_lags=2)
    noise = evaluate.summarize(evaluate.per_date_metrics(_synthetic_eval_panel(0.0), "sig"), overlap_lags=2)
    assert strong["mean_rank_ic"] > 0.3 and strong["rank_ic_tstat_nw"] > 5 and strong["mean_spread"] > 0
    assert abs(noise["mean_rank_ic"]) < 0.1 and abs(noise["rank_ic_tstat_nw"]) < 3


def test_dates_with_too_few_names_are_skipped():
    panel = _synthetic_eval_panel(0.5, names=5)
    assert evaluate.per_date_metrics(panel, "sig").empty


def test_newey_west_with_zero_lags_is_the_naive_tstat():
    x = np.random.default_rng(3).normal(0.1, 1, 200)
    naive = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert evaluate.newey_west_tstat(x.tolist(), lags=0) == pytest.approx(naive)


def test_overlap_lags():
    assert evaluate.overlap_lags(63, 21) == 2
    assert evaluate.overlap_lags(21, 21) == 0


def test_walk_forward_splits_are_ordered_embargoed_and_disjoint():
    dates = pd.bdate_range("2020-01-01", periods=30)
    splits = list(evaluate.walk_forward_splits(dates, min_train=10, test_size=5, embargo=3))
    assert splits
    for train, test in splits:
        gap = dates.get_loc(test[0]) - dates.get_loc(train[-1])
        assert gap == 3 + 1  # 3 embargoed dates between last train and first test
        assert max(train) < min(test)
    tests = [d for _, t in splits for d in t]
    assert len(tests) == len(set(tests))


def test_label_buckets_order_matches_app_labels():
    panel = pd.DataFrame({
        "score_full": [0.5, 0.4, 0.2, 0.0, -0.3, None],
        "fwd_excess": [0.10, 0.05, 0.02, 0.0, -0.05, 0.01],
    })
    buckets = evaluate.label_buckets(panel)
    assert list(buckets.index) == ["Strong", "Favorable", "Neutral", "Unfavorable", "Insufficient data"]
    assert buckets.loc["Strong", "count"] == 2
