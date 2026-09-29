"""PHASE 2 engine and signal tests: timing, look-ahead, costs, periods, lock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import events as ev
from backtest import phase2_signals as sig


def _calendar(start="2020-01-01", n=400):
    return pd.bdate_range(start, periods=n)


def _ohlc(calendar, start=100.0, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + drift + rng.normal(0, 0.005, len(calendar)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                         "low": np.minimum(open_, close) * 0.998, "close": close}, index=calendar)


def _flat(calendar, price=100.0):
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price}, index=calendar, dtype=float)


def test_entry_is_next_open_after_filing_day():
    cal = _calendar("2021-01-04", 30)  # Mon
    friday = pd.Timestamp("2021-01-08")
    saturday = pd.Timestamp("2021-01-09")
    assert cal[ev.signal_positions(cal, [friday])[0] + 1] == pd.Timestamp("2021-01-11")
    assert cal[ev.signal_positions(cal, [saturday])[0] + 1] == pd.Timestamp("2021-01-11")


def test_time_exit_return_costs_and_excess():
    cal = _calendar(n=60)
    stock = _flat(cal)
    stock.loc[cal[5]:, ["open", "high", "low", "close"]] = 110.0  # jumps before entry? no: entry at pos 3
    stock.iloc[:5] = 100.0
    bench = _flat(cal)
    events = pd.DataFrame({"signal_date": [cal[2]], "ticker": ["AAA"], "strength": [1.0]})
    rule = ev.BarrierExit(hold=10, stop_atr=None, cost=0.003)
    trades = ev.build_trades(events, {"AAA": stock}, bench, rule)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_date"] == cal[3] and t["entry_price"] == 100.0
    assert t["exit_date"] == cal[12] and t["exit_price"] == 110.0
    assert t["ret"] == pytest.approx(110 * 0.997 / (100 * 1.003) - 1)
    assert t["excess"] == pytest.approx(t["ret"] - 0.0)


def test_stop_exits_and_gap_fills_at_open():
    cal = _calendar(n=60)
    stock = _ohlc(cal, seed=1)
    levels = pd.DataFrame({"close": stock["close"], "atr_14d": 1.0}, index=cal)
    sp = 20
    stop_level = stock["close"].iloc[sp] - 2.5
    stock.iloc[sp + 3, stock.columns.get_loc("open")] = stop_level - 5  # gap below the stop
    events = pd.DataFrame({"signal_date": [cal[sp]], "ticker": ["AAA"], "strength": [1.0]})
    trades = ev.build_trades(events, {"AAA": stock}, _flat(cal), ev.BarrierExit(63, 2.5, 0.0), {"AAA": levels})
    t = trades.iloc[0]
    assert t["outcome"] == "stop"
    assert t["exit_pos"] <= sp + 3
    if t["exit_pos"] == sp + 3:
        assert t["exit_price"] == pytest.approx(stop_level - 5)


def test_development_trades_must_exit_before_holdout():
    trades = pd.DataFrame({
        "signal_date": pd.to_datetime(["2024-06-01", "2024-08-20", "2024-09-02"]),
        "exit_date": pd.to_datetime(["2024-08-01", "2024-10-01", "2024-12-01"]),
        "excess": [0.1, 0.2, 0.3],
    })
    dev = ev.period_trades(trades, "development")
    hold = ev.period_trades(trades, "holdout")
    assert list(dev["excess"]) == [0.1]
    assert list(hold["excess"]) == [0.3]


def test_eligibility_never_uses_a_future_membership_list():
    members = pd.DataFrame({"date": pd.to_datetime(["2021-01-29", "2021-02-26"]), "ticker": ["OLD", "NEW"]})
    e = ev.Eligibility(members)
    assert e.pool("2021-02-25") == ["OLD"]
    assert e.pool("2021-02-26") == ["NEW"]
    assert e.pool("2021-01-01") == []
    assert not e.contains("2021-02-10", "NEW")


def test_random_baseline_is_reproducible_and_eligible():
    members = pd.DataFrame({"date": pd.Timestamp("2021-01-29"), "ticker": ["A", "B", "C", "D"]})
    e = ev.Eligibility(members)
    trades = pd.DataFrame({"signal_date": pd.to_datetime(["2021-02-03"] * 20)})
    a, b = ev.random_events(trades, e, seed=7), ev.random_events(trades, e, seed=7)
    assert a.equals(b)
    assert set(a["ticker"]) <= {"A", "B", "C", "D"}


def test_portfolio_single_trade_and_slot_limit():
    cal = _calendar(n=40)
    stock = _flat(cal)
    stock.loc[cal[10]:, ["open", "high", "low", "close"]] = 120.0
    trades = pd.DataFrame({"signal_date": [cal[1]], "ticker": ["A"], "strength": [1.0], "entry_pos": [2],
                           "exit_pos": [20], "entry_price": [100.0], "exit_price": [120.0]})
    sim = ev.simulate_portfolio(trades, {"A": stock}, cal, cost=0.0, max_open=4)
    assert sim["equity"].iloc[-1] == pytest.approx(1 + 0.25 * 0.20)
    many = pd.DataFrame({"signal_date": cal[1], "ticker": [f"T{i}" for i in range(6)], "strength": range(6),
                         "entry_pos": 2, "exit_pos": 20, "entry_price": 100.0, "exit_price": 100.0})
    sim = ev.simulate_portfolio(many, {f"T{i}": _flat(cal) for i in range(6)}, cal, cost=0.0, max_open=4)
    assert len(sim["taken"]) == 4
    assert set(sim["taken"]["ticker"]) == {"T5", "T4", "T3", "T2"}  # strongest first


def test_holdout_lock_allows_one_run(tmp_path):
    ev.holdout_lock(tmp_path, "abc", force=False)
    with pytest.raises(RuntimeError):
        ev.holdout_lock(tmp_path, "abc", force=False)
    record = ev.holdout_lock(tmp_path, "abc", force=True)
    assert record["runs"][-1]["forced"] is True


def test_verdict_needs_both_periods():
    ok, bad = {"passed": True}, {"passed": False}
    assert ev.verdict(ok, ok) == "PASS"
    assert ev.verdict(ok, bad) == "NOT PROVEN"
    assert ev.verdict(bad, ok) == "NOT PROVEN"
    assert ev.verdict(ok, None).startswith("DEVELOPMENT")


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=600)
    trades = pd.DataFrame({"entry_date": rng.choice(dates, 500), "excess": rng.normal(0.02, 0.05, 500)})
    lo, hi = ev.bootstrap_ci(trades)
    assert lo < trades["excess"].mean() < hi


# ---------------------------------------------------------------- signals


def _tx(rows):
    return pd.DataFrame([{"accession": f"a{i}", "issuer_cik": 1, "owner_cik": o, "trans_code": code,
                          "filing_date": d, "trans_date": d, "shares": 100, "price": 10.0, "value": 1000.0}
                         for i, (o, d, code) in enumerate(rows)])


def test_h1_fires_on_the_third_buyers_filing_day_only():
    tx = _tx([(11, "2020-03-02", "P"), (12, "2020-03-10", "P"), (13, "2020-03-20", "P"), (14, "2020-03-25", "P"),
              (15, "2020-03-21", "S")])
    events = sig.insider_cluster_events(tx, {1: "AAA"})
    assert list(events["signal_date"]) == [pd.Timestamp("2020-03-20")]  # 4th buyer is inside the cooldown


def test_h1_ignores_buyers_spread_beyond_the_window_and_sales():
    tx = _tx([(11, "2020-01-02", "P"), (12, "2020-01-20", "P"), (13, "2020-02-15", "P"),
              (14, "2020-02-16", "S"), (15, "2020-02-16", "S")])
    assert sig.insider_cluster_events(tx, {1: "AAA"}).empty


def test_h1_cooldown_expires():
    tx = _tx([(11, "2020-01-02", "P"), (12, "2020-01-03", "P"), (13, "2020-01-06", "P"),
              (11, "2020-06-01", "P"), (12, "2020-06-02", "P"), (13, "2020-06-03", "P")])
    events = sig.insider_cluster_events(tx, {1: "AAA"})
    assert list(events["signal_date"]) == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-06-03")]


def test_h2_threshold_uses_only_earlier_filings():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2019-01-01", periods=300)
    sue = {f"T{i}": pd.DataFrame({"filed": [days[i]], "sue": [rng.normal()]}) for i in range(300)}
    base = sig.earnings_surprise_events(sue, lambda d, t: True, min_history=50)
    later = dict(sue)
    later["BIG"] = pd.DataFrame({"filed": [days[-1] + pd.Timedelta(days=1)], "sue": [1e6]})
    with_future = sig.earnings_surprise_events(later, lambda d, t: True, min_history=50)
    past = with_future[with_future["ticker"] != "BIG"].reset_index(drop=True)
    pd.testing.assert_frame_equal(base, past)
    assert "BIG" in set(with_future["ticker"])


def test_h2_needs_enough_history():
    days = pd.bdate_range("2019-01-01", periods=20)
    sue = {f"T{i}": pd.DataFrame({"filed": [days[i]], "sue": [float(i)]}) for i in range(20)}
    assert sig.earnings_surprise_events(sue, lambda d, t: True).empty


def test_h3_trend_filter_ignores_later_prices():
    cal = pd.bdate_range("2019-01-01", periods=400)
    up = pd.DataFrame({"adj_close": np.linspace(50, 150, 400)}, index=cal)
    d = cal[300]
    members = pd.DataFrame({"date": [d], "ticker": ["AAA"], "is_candidate": [True], "drawdown": [0.3]})
    first = sig.cheap_quality_trend_events(members, {"AAA": up})
    crash = up.copy()
    crash.iloc[301:] = 1.0
    second = sig.cheap_quality_trend_events(members, {"AAA": crash})
    assert len(first) == 1
    pd.testing.assert_frame_equal(first, second)
