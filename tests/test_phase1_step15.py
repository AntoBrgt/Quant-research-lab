"""PHASE 1 / STEP 15: XBRL point-in-time rules, survival labels and censoring. Synthetic, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sec_xbrl as sx
from backtest import value_survival as vs

CAL = pd.bdate_range("2020-01-01", periods=400)


# ---------------------------------------------------------------- XBRL


def _doc(facts: dict) -> dict:
    """facts: {tag: [(start, end, val, filed, form)]} -- start None for instants."""
    usgaap = {}
    for tag, rows in facts.items():
        usgaap[tag] = {"units": {"USD": [
            {**({"start": s} if s else {}), "end": e, "val": v, "filed": f, "form": form, "accn": f"a{i}"}
            for i, (s, e, v, f, form) in enumerate(rows)]}}
    return {"facts": {"us-gaap": usgaap}}


def test_ytd_cash_flow_is_differenced_into_quarters_available_from_the_later_filing():
    doc = _doc({"NetCashProvidedByUsedInOperatingActivities": [
        ("2021-01-01", "2021-03-31", 10, "2021-05-01", "10-Q"),
        ("2021-01-01", "2021-06-30", 25, "2021-08-01", "10-Q"),
        ("2021-01-01", "2021-09-30", 45, "2021-11-01", "10-Q"),
        ("2021-01-01", "2021-12-31", 70, "2022-02-15", "10-K"),
    ]})
    q = sx.discrete_quarters(sx.duration_facts(doc, sx.DURATION_TAGS["ocf"])).set_index("end")
    assert q["val"].tolist() == [10, 15, 20, 25]
    assert q.loc[pd.Timestamp("2021-12-31"), "filed"] == pd.Timestamp("2022-02-15")


def test_earliest_filed_value_wins_across_tag_switches():
    doc = _doc({
        "NetCashProvidedByUsedInOperatingActivities": [("2020-01-01", "2020-12-31", 999, "2022-02-15", "10-K")],  # comparative, restated
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": [("2020-01-01", "2020-12-31", 100, "2021-02-15", "10-K")],
    })
    f = sx.duration_facts(doc, sx.DURATION_TAGS["ocf"])
    row = f.iloc[0]
    assert row["val"] == 100 and row["filed"] == pd.Timestamp("2021-02-15")


def _table(rows):
    """rows: (kind, field, end, val, filed)."""
    return pd.DataFrame([{"kind": k, "field": f, "end": pd.Timestamp(e), "val": float(v), "filed": pd.Timestamp(fd), "derived": False}
                         for k, f, e, v, fd in rows])


def test_snapshot_only_sees_values_filed_by_the_as_of_date():
    rows = [("q", "revenue", e, 100, fd) for e, fd in
            [("2020-03-31", "2020-05-01"), ("2020-06-30", "2020-08-01"), ("2020-09-30", "2020-11-01"), ("2020-12-31", "2021-02-20")]]
    table = _table(rows)
    assert sx.snapshot(table, "2021-02-19")["revenue_ttm"] is None  # Q4 period ended, not filed yet
    assert sx.snapshot(table, "2021-02-20")["revenue_ttm"] == 400


def test_sue_is_available_from_the_filing_date():
    ends = pd.date_range("2016-03-31", periods=20, freq="QE")
    rng = np.random.default_rng(0)
    rows = [("q", "eps", e, 1 + 0.1 * i + rng.normal(0, 0.05), e + pd.Timedelta(days=40)) for i, e in enumerate(ends)]
    sue = sx.sue_series(_table(rows))
    assert len(sue) > 0
    assert (sue["filed"] - sue["end"]).dt.days.eq(40).all()  # dated by filing, never by period end


def test_thesis_break_is_known_on_the_second_bad_filing():
    rows = []
    for i, e in enumerate(pd.date_range("2019-03-31", periods=12, freq="QE")):
        bad = i >= 9  # last 3 quarters: negative FCF and revenue below a year earlier
        rows += [("q", "revenue", e, 80 if bad else 100 + i, e + pd.Timedelta(days=35)),
                 ("q", "ocf", e, -5 if bad else 10, e + pd.Timedelta(days=35)),
                 ("q", "capex", e, 1, e + pd.Timedelta(days=35))]
    breaks = sx.thesis_break_dates(_table(rows))
    second_bad = pd.date_range("2019-03-31", periods=12, freq="QE")[10] + pd.Timedelta(days=35)
    assert breaks[0] == second_bad


# ---------------------------------------------------------------- outcomes and censoring


def _ohlc(highs, opens=None, closes=None):
    n = len(highs)
    opens = opens if opens is not None else [100.0] * n
    closes = closes if closes is not None else [100.0] * n
    frame = pd.DataFrame({"open": opens, "high": highs, "low": [90.0] * n, "close": closes}, index=CAL[:n])
    return frame.reindex(CAL)


def test_target_hit_is_event_and_entry_is_next_open():
    highs = [100] * 5 + [131] + [100] * 20
    out = vs.trade_outcome(_ohlc(highs), CAL[0], targets=(0.30,), cost=0.0)
    assert out["entry_date"] == CAL[1] and out["entry_price"] == 100
    assert out["t30_event"] == 1 and out["t30_reason"] == "target" and out["t30_exit_price"] == pytest.approx(130)
    assert out["t30_duration"] == 5  # entry day = day 1


def test_thesis_break_exits_next_open_and_is_not_an_event():
    highs = [100] * 30
    opens = [100] * 10 + [70] + [100] * 19
    out = vs.trade_outcome(_ohlc(highs, opens=opens), CAL[0], targets=(0.30,), break_dates=[CAL[9]], cost=0.0)
    assert out["t30_event"] == 0 and out["t30_reason"] == "thesis_break"
    assert out["t30_exit_date"] == CAL[10] and out["t30_exit_price"] == 70


def test_data_end_is_censored_not_a_failure():
    out = vs.trade_outcome(_ohlc([100] * 30), CAL[0], targets=(0.30,), cost=0.0)
    assert out["t30_event"] == 0 and out["t30_reason"] == "data_end"


def test_training_labels_are_censored_at_the_cutoff():
    # Entered at day 1, hit the target on day 200 -- a model trained at day 100 must see "still open", not a failure.
    frame = pd.DataFrame([{"entry_date": CAL[1], "t30_exit_date": CAL[200], "t30_event": 1, "t30_duration": 200,
                           "t30_reason": "target", "t30_return": 0.3}])
    censored = vs.censor_at(frame, "t30", CAL[100], CAL)
    row = censored.iloc[0]
    assert row["t30_event"] == 0 and row["t30_reason"] == "open_at_cutoff" and row["t30_duration"] == 100
    # Positions entered after the cutoff don't exist yet for that model.
    later = frame.assign(entry_date=CAL[150])
    assert vs.censor_at(later, "t30", CAL[100], CAL).empty


def test_expected_annualized_return_uses_fail_return_and_curve():
    always = vs.expected_annualized_return(lambda t: np.array([1.0 if t < 252 else 0.0]), 0.30, -0.20)
    never = vs.expected_annualized_return(lambda t: np.array([1.0]), 0.30, -0.20)
    assert always["expected_return"][0] == pytest.approx(0.30)
    assert never["expected_return"][0] == pytest.approx(-0.20)
    assert always["p_hit_24m"][0] == 1.0 and always["expected_annualized_return"][0] > never["expected_annualized_return"][0]


def test_portfolio_verdict_needs_every_condition():
    idx = pd.bdate_range("2015-01-01", periods=2520)
    spy = pd.Series(100 * 1.0003 ** np.arange(2520), index=idx)
    noise = np.random.default_rng(0).normal(0, 0.01, 2520)
    spy = spy * np.cumprod(1 + noise)  # SPY: +7.5%/yr drift, 16% vol
    strong = pd.Series(np.cumprod(1 + 0.0006 + 0.5 * noise), index=idx)  # higher drift, half the vol
    verdict, lines, _ = vs.portfolio_verdict(strong, spy, [0.0] * 100)
    assert verdict == "BEATS MARKET", lines
    lagging_random = vs.portfolio_verdict(strong, spy, [1.0] * 100)  # every random portfolio did better
    assert lagging_random[0] == "DOES NOT BEAT MARKET"
    weak = pd.Series(np.cumprod(1 + 0.00005 + noise), index=idx)  # same risk as SPY, less drift
    assert vs.portfolio_verdict(weak, spy, [0.0] * 100)[0] == "DOES NOT BEAT MARKET"
