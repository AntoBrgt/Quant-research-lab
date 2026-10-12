"""PHASE 1b: filing sections, Lazy Prices features, panel transforms, ranker, vol targeting. Synthetic, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import edgar_sections as es
from backtest import lazy_prices as lp
from backtest import phase1b as p1b
from backtest import protocol

FILLER = "The company faces competition and regulatory risk in its markets. " * 40  # > MIN_SECTION_CHARS


# ---------------------------------------------------------------- section extraction


def _doc(lines):
    return "\n".join(lines)


def test_sections_skip_toc_and_running_headers_and_join_split_headings():
    text = _doc([
        "Table of Contents",
        "Item 1A.", "Risk Factors", "12",          # TOC row: heading then page number
        "Item 7.", "Management's Discussion and Analysis", "30",
        "Item 1. Business", FILLER,
        "Item 1A.", "Risk Factors",                 # body heading split over two lines (older filings)
        FILLER, "Item 1A", FILLER,                  # running page header repeating the same item: not an end
        "Item 1B. Unresolved Staff Comments", "None.",
        "Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations",
        FILLER, "Item 7A. Quantitative and Qualitative Disclosures About Market Risk", "Rates.",
    ])
    sections = es.extract_sections(text, "10-K")
    assert sections["risk_factors"].count("competition") == 80  # both FILLER blocks, across the running header
    assert "Unresolved" not in sections["risk_factors"]
    assert sections["mdna"].strip().startswith("The company") and "Rates." not in sections["mdna"]


def test_10q_mdna_is_item_2_and_short_stubs_are_dropped():
    text = _doc(["Item 2. Management's Discussion and Analysis of Financial Condition", FILLER,
                 "Item 3. Quantitative and Qualitative Disclosures", "x",
                 "Item 1A. Risk Factors", "There have been no material changes.", "Item 2. Unregistered Sales", "y"])
    sections = es.extract_sections(text, "10-Q")
    assert sections["mdna"] and sections["risk_factors"] is None  # "no material changes" is a stub, not a section


def test_decode_falls_back_to_windows_1252():
    assert es.decode("Company’s".encode("cp1252")) == "Company’s"
    assert es.decode("Company’s".encode("utf-8")) == "Company’s"


# ---------------------------------------------------------------- Lazy Prices


def _filings():
    rows = []
    for year in (2013, 2014, 2015):
        rows.append({"cik": 1, "accession": f"K{year}", "form": "10-K", "filing_date": f"{year}-02-20", "report_date": f"{year - 1}-12-31"})
        for q, month in ((1, "05"), (2, "08"), (3, "11")):
            rows.append({"cik": 1, "accession": f"Q{year}{q}", "form": "10-Q", "filing_date": f"{year}-{month}-05",
                         "report_date": f"{year}-{int(month) - 2:02d}-30"})
    frame = pd.DataFrame(rows)
    frame["risk_factors"] = FILLER
    frame["mdna"] = FILLER
    frame.loc[frame["accession"] == "K2015", "risk_factors"] = "Brand new cyber security pandemic litigation exposure. " * 40
    return frame


def test_filings_pair_with_the_same_form_a_year_earlier():
    paired = lp.pair_filings(_filings()).set_index("accession")
    assert paired.at["K2014", "prior_accession"] == "K2013"
    assert paired.at["Q20152", "prior_accession"] == "Q20142"
    assert pd.isna(paired.at["K2013", "prior_accession"])


def test_similarity_drops_when_the_text_is_rewritten():
    filings = _filings()
    rewritten = filings.loc[filings["accession"] == "K2015", "risk_factors"].iloc[0]
    other = "Interest rate hedging and foreign exchange derivative contracts. " * 40
    vec = lp.fit_tfidf([FILLER] * 6 + [rewritten] * 6 + [other] * 6)  # terms spread across docs, like a real corpus
    feats = lp.filing_features(filings, vec, use_embeddings=False).set_index("accession")
    assert feats.at["K2014", "sim_tfidf_rf"] == pytest.approx(1.0, abs=1e-6)
    assert feats.at["K2015", "sim_tfidf_rf"] < 0.5
    assert feats.at["K2015", "dsim_tfidf_rf"] < -0.5  # vs the previous 10-K comparison
    assert feats.at["K2014", "sim_tfidf_mdna"] == pytest.approx(1.0, abs=1e-6)


def test_monthly_text_features_exist_only_from_the_filing_date_and_expire():
    per_filing = pd.DataFrame({"cik": [1], "filing_date": pd.to_datetime(["2015-02-20"]),
                               **{f: [0.5] for f in lp.FEATURES}})
    keys = pd.DataFrame({"date": pd.to_datetime(["2015-01-30", "2015-02-27", "2015-12-31"]), "ticker": "AAA"})
    out = lp.monthly_features(per_filing, keys, {"AAA": 1})
    assert out["sim_tfidf_rf"].isna().tolist() == [True, False, True]  # before filing: none; >200 days later: stale


# ---------------------------------------------------------------- panel transforms


def test_industry_relative_and_minimum_group_size():
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-31"] * 4), "x": [1.0, 2.0, 3.0, 10.0]})
    rel = p1b.industry_relative(panel, ["x"], pd.Series([1, 1, 1, 2]))
    assert rel["x_ind"].tolist()[:3] == [-1.0, 0.0, 1.0] and np.isnan(rel["x_ind"].iloc[3])


def test_changes_use_exact_calendar_months():
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-31", "2020-02-28", "2020-04-30"]), "ticker": "A", "x": [1.0, 3.0, 7.0]})
    out = p1b.changes(panel, ["x"], 1)
    assert np.isnan(out["x_chg1m"].iloc[0]) and out["x_chg1m"].iloc[1] == 2.0 and np.isnan(out["x_chg1m"].iloc[2])  # March missing


def test_vol_targeting_uses_only_past_returns():
    idx = pd.date_range("2016-01-31", periods=30, freq="ME")
    r = pd.Series(np.random.default_rng(0).normal(0.01, 0.05, 30), index=idx)
    cash = pd.Series(0.001, index=idx)
    base = p1b.vol_targeted(r, cash)
    shocked = r.copy()
    shocked.iloc[20] = 0.5  # a huge month must not change that same month's exposure
    after = p1b.vol_targeted(shocked, cash)
    exposure = lambda out, ret: (out - cash) / (ret - cash)  # noqa: E731
    assert exposure(base, r).iloc[20] == pytest.approx(exposure(after, shocked).iloc[20])
    assert (exposure(base, r).dropna() <= p1b.MAX_LEVERAGE + 1e-9).all()


def test_lambdarank_learns_a_planted_signal_out_of_sample():
    rng = np.random.default_rng(1)
    rows = []
    for d in pd.date_range("2014-01-31", periods=60, freq="ME"):
        sig = rng.normal(size=80)
        ret = 0.5 * sig + rng.normal(size=80)
        for i in range(80):
            rows.append({"date": d, "ticker": f"T{i}", "momentum_12_1": sig[i], "return_20d": rng.normal(), "fwd_excess": ret[i]})
    panel = pd.DataFrame(rows)
    score = p1b.walk_forward_rank(panel, ["momentum_12_1", "return_20d"], num_trees=50)
    oos = score.notna()
    assert oos.any() and score[~oos.values].isna().all()
    ic = panel[oos].assign(s=score[oos]).groupby("date").apply(lambda g: g["s"].corr(g["fwd_excess"], method="spearman")).mean()
    assert ic > 0.25


def test_next_month_labels_never_reach_into_the_holdout(monkeypatch):
    idx = pd.bdate_range("2023-10-02", "2024-02-28")
    full = pd.DataFrame({"open": np.linspace(100, 110, len(idx)), "adj_close": 1.0}, index=idx)
    monkeypatch.setattr(protocol.data, "load_ext_prices", lambda t: full)
    panel = pd.DataFrame({"date": pd.to_datetime(["2023-10-31", "2023-11-30", "2023-12-29"]), "ticker": "AAA"})
    ret, exits = p1b.next_month_returns(panel)
    assert ret.notna().tolist() == [True, False, False]  # Nov -> Dec label would exit in January 2024
    assert exits.dropna().max() < protocol.HOLDOUT_START
