"""STEP 13: broad PIT universe, best-ideas 13F features, and the tightened promotion rule. Synthetic, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import dataset, model
from institutional_research import holdings_13f as h13f


# ---------------------------------------------------------------- pit_broad


def _history(rows):
    """Selection history rows: (cik, institution, accession, filing_date, report_period, ticker, direction)."""
    return pd.DataFrame(
        [{"cik": c, "institution": i, "accession": a, "filing_date": f, "report_period": r,
          "ticker": t, "cusip": f"CU_{t}", "view_direction": d} for c, i, a, f, r, t, d in rows]
    )


HOLDINGS = {
    # accession -> {cusip: value}; pre-2023 filings report values in thousands
    "F1-Q4": {"CU_AAA": 500, "CU_BBB": 300, "CU_ETF": 900, "CU_GONE": 250, "CU_SMALL": 1},
    "F2-Q4": {"CU_AAA": 100, "CU_CCC": 350},
    "F1-Q1": {"CU_AAA": 500_000, "CU_DDD": 2_000_000},  # filed after the 2023 switch: dollars
    "F2-Q1": {"CU_CCC": 400_000},
}
RESOLUTIONS = {
    "CU_AAA": {"ticker": "AAA", "security_type": "Common Stock"},
    "CU_BBB": {"ticker": "BBB", "security_type": "Common Stock"},
    "CU_CCC": {"ticker": "CCC", "security_type": "Common Stock"},
    "CU_DDD": {"ticker": "DDD", "security_type": "Common Stock"},
    "CU_SMALL": {"ticker": "SMALL", "security_type": "Common Stock"},
    "CU_ETF": {"ticker": "SPY", "security_type": "ETP"},
    "CU_GONE": None,  # delisted: OpenFIGI can't map it
}


def _loader(filing):
    values = HOLDINGS[filing.accession]
    return pd.DataFrame({"cusip": list(values), "value": [float(v) for v in values.values()]})


def _broad(dates, top_n=3):
    history = _history([
        (1, "F1", "F1-Q4", "2022-11-14", "2022-09-30", "AAA", "POSITIVE"),
        (2, "F2", "F2-Q4", "2022-11-10", "2022-09-30", "CCC", "NEGATIVE"),
        (1, "F1", "F1-Q1", "2023-02-14", "2022-12-31", "DDD", "POSITIVE"),
        (2, "F2", "F2-Q1", "2023-02-10", "2022-12-31", "CCC", "MENTIONED"),
    ])
    membership = dataset.pit_broad_universe(history, pd.to_datetime(dates), top_n=top_n,
                                            holdings_loader=_loader, resolve=lambda cs: {c: RESOLUTIONS.get(c) for c in cs})
    return membership.set_index(["date", "ticker"], drop=False)


def test_broad_universe_ranks_total_value_across_filers_and_excludes_funds():
    m = _broad(["2022-12-01"])
    day = m.loc[pd.Timestamp("2022-12-01")]
    # AAA 600 (two filers), CCC 350, BBB 300, GONE 250; the ETF (900) is excluded.
    assert day["ticker"].tolist() == ["AAA", "CCC", "BBB"]
    assert day.loc[day["ticker"] == "AAA", "institution_count"].iloc[0] == 2
    assert "SPY" not in set(day["ticker"])


def test_broad_universe_keeps_unresolved_members_for_coverage():
    m = _broad(["2022-12-01"], top_n=4)
    day = m.reset_index(drop=True)
    gone = day[day["cusip"] == "CU_GONE"]
    assert len(gone) == 1 and gone["ticker"].isna().all() and gone["institutional_direction_score"].isna().all()


def test_broad_universe_uses_only_filings_public_on_the_date():
    before = _broad(["2023-02-12"]).reset_index(drop=True)
    after = _broad(["2023-02-15"]).reset_index(drop=True)
    assert "DDD" not in set(before["ticker"])  # F1's Q1 filing (2023-02-14) is not public yet
    assert "DDD" in set(after["ticker"])


def test_broad_universe_normalizes_13f_value_units_across_the_2023_switch():
    # On 2023-02-12: F1 still on its 2022 filing (thousands), F2 on its 2023 filing (dollars).
    day = _broad(["2023-02-12"]).reset_index(drop=True).set_index("ticker")
    assert day.loc["AAA", "total_13f_value"] == pytest.approx(500_000)  # 500 thousand, not 500
    assert day.loc["CCC", "total_13f_value"] == pytest.approx(400_000)
    assert day.index[0] == "AAA"


def test_broad_direction_score_comes_from_selected_rows_and_defaults_to_zero():
    day = _broad(["2022-12-01"]).reset_index(drop=True).set_index("ticker")
    assert day.loc["AAA", "institutional_direction_score"] == 1.0
    assert day.loc["CCC", "institutional_direction_score"] == -1.0
    assert day.loc["BBB", "institutional_direction_score"] == 0.0


def test_pit_broad_panel_requires_a_membership():
    bench = pd.DataFrame({"adj_close": np.linspace(100, 120, 400)}, index=pd.bdate_range("2021-01-01", periods=400))
    with pytest.raises(ValueError, match="pit_broad"):
        dataset.build_panel(None, price_loader=lambda t: None, fundamentals_loader=None, benchmark=bench, universe="pit_broad")


# ---------------------------------------------------------------- active managers


def _positions(rows):
    return pd.DataFrame([
        {"filing_date": f, "report_period": None, "institution": "Berkshire", "cik": 1067983, "accession": a,
         "cusip": f"CU_{t}", "ticker": t, "status": s, "weight": w, "adjusted_change": None}
        for a, f, t, s, w in rows
    ], columns=h13f.POSITION_HISTORY_COLUMNS)


def test_active_features_point_in_time_and_zero_vs_nan():
    positions = _positions([
        ("B1", "2023-02-14", "AAA", "UNCHANGED", 0.40), ("B1", "2023-02-14", "BBB", "NEW", 0.02),
        ("B2", "2023-05-15", "AAA", "REDUCED", 0.35), ("B2", "2023-05-15", "CCC", "ADDED", 0.05),
    ])
    keys = pd.DataFrame({"date": pd.to_datetime(["2023-01-10", "2023-03-01", "2023-03-01", "2023-03-01", "2023-06-01", "2023-06-01"]),
                         "ticker": ["AAA", "AAA", "BBB", "ZZZ", "BBB", "CCC"]})
    out = dataset.active_manager_features(keys, positions).set_index(["date", "ticker"])
    assert out.loc[(pd.Timestamp("2023-01-10"), "AAA")].isna().all()  # no manager filing public yet: unknown
    assert out.loc[(pd.Timestamp("2023-03-01"), "AAA"), "active_weight_max"] == 0.40
    assert out.loc[(pd.Timestamp("2023-03-01"), "AAA"), "active_new_or_add"] == 0
    assert out.loc[(pd.Timestamp("2023-03-01"), "BBB"), "active_new_or_add"] == 1
    assert out.loc[(pd.Timestamp("2023-03-01"), "ZZZ")].tolist() == [0.0, 0.0]  # not held = 0
    # After B2 is filed, BBB is no longer in the latest filing, CCC was added.
    assert out.loc[(pd.Timestamp("2023-06-01"), "BBB")].tolist() == [0.0, 0.0]
    assert out.loc[(pd.Timestamp("2023-06-01"), "CCC"), "active_new_or_add"] == 1


def test_active_managers_default_and_csv(tmp_path):
    assert h13f.load_active_managers(tmp_path / "missing.csv") == {"Berkshire": 1067983}
    path = tmp_path / "managers.csv"
    path.write_text("institution,cik\nPershing Square,1336528\n")
    assert h13f.load_active_managers(path) == {"Pershing Square": 1336528}


def test_log_market_cap_and_new_features_are_in_the_whitelist():
    for feature in ["log_market_cap", "insider_buyers_90d", "insider_net_buy_value_90d_mcap", "insider_cluster_buy",
                    "active_new_or_add", "active_weight_max", "institution_count"]:
        assert feature in model.FEATURES


# ---------------------------------------------------------------- promotion rule


def _per_date(ics, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(ics), freq="21B")
    return pd.DataFrame({"date": dates, "rank_ic": ics})


def test_promotion_needs_significant_ic_difference_not_just_a_higher_mean():
    rng = np.random.default_rng(0)
    n = 60
    base = rng.normal(0.02, 0.10, n)
    # Slightly higher on average but very noisy relative to the baseline: mean beats, difference t < 2.
    ml = base + rng.normal(0.005, 0.10, n)
    lines = model.promotion_check({"score_ml": _per_date(ml), "score_rank": _per_date(base), "momentum_12_1": _per_date(base)}, lags=2)
    assert any(line.startswith("- FAIL -- beats score_rank") for line in lines)
    assert "does NOT qualify" in lines[-1]


def test_promotion_passes_a_consistently_better_model_over_enough_years():
    rng = np.random.default_rng(1)
    n = 60  # ~5 years of monthly dates
    base = rng.normal(0.0, 0.05, n)
    ml = base + 0.05 + rng.normal(0, 0.01, n)
    lines = model.promotion_check({"score_ml": _per_date(ml), "score_rank": _per_date(base), "momentum_12_1": _per_date(base)}, lags=2)
    assert "qualifies for promotion" in lines[-1], lines


def test_promotion_requires_at_least_four_oos_years():
    ml = _per_date([0.2] * 30)  # 30 monthly dates = ~2.4 years
    base = _per_date([0.0] * 30)
    lines = model.promotion_check({"score_ml": ml, "score_rank": base, "momentum_12_1": base}, lags=0)
    assert any(line.startswith("- FAIL -- >= 4 OOS years") for line in lines)
