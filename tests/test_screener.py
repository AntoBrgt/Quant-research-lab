"""screener.rank_universe -- offline, with fake price/fundamental providers."""

from __future__ import annotations

import pandas as pd

import research_engine
import screener


def _prices(step, n=300, start=100.0):
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    closes = [max(1.0, start + i * step) for i in range(len(dates))]
    df = pd.DataFrame(
        {"open": closes, "adj_close": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes], "volume": [1_000_000.0] * len(dates)},
        index=dates,
    )
    df.index.name = "date"
    return df


class _Prices:
    def __init__(self, by_ticker):
        self.by_ticker = by_ticker

    def get_price_history(self, ticker):
        if ticker not in self.by_ticker:
            raise ValueError(f"No price data available for {ticker}")
        return self.by_ticker[ticker]


class _Fundamentals:
    def get_raw_fundamentals(self, ticker):
        return {"info": {"sector": "Technology"}, "income_stmt": {}, "cashflow": {}, "balance_sheet": {}}


def _loader(prices):
    def load(ticker, horizon_days, institutional_mentions=None):
        return research_engine.load_extended_research(
            ticker, horizon_days, institutional_mentions=institutional_mentions,
            price_prov=prices, fundamentals_prov=_Fundamentals(), benchmark_prov=_Prices({"SPY": _prices(0.05)}),
        )
    return load


def _universe(tickers):
    return pd.DataFrame(
        [{"ticker": t, "company_name": f"{t} Inc", "sector": None, "institution_count": 1, "institutional_direction_score": 0.0} for t in tickers]
    )


def test_uptrend_ranks_above_downtrend_and_failures_sink_to_bottom():
    prices = _Prices({"UP": _prices(0.5), "DOWN": _prices(-0.25)})
    ranking = screener.rank_universe(_universe(["DOWN", "BROKEN", "UP"]), 3, load=_loader(prices), max_workers=2)

    assert ranking["ticker"].tolist()[0] == "UP"
    assert ranking["ticker"].tolist()[-1] == "BROKEN"
    assert not ranking.set_index("ticker").loc["BROKEN", "shortlisted"]  # no data -> never shortlisted
    assert ranking["rank"].tolist() == [1, 2, 3]
    assert list(ranking.columns) == screener.RANKING_COLUMNS
    up = ranking.set_index("ticker").loc["UP"]
    assert up["label"] in ("Strong", "Favorable")
    assert up["horizon_score"] > ranking.set_index("ticker").loc["DOWN", "horizon_score"]


def test_institutional_tilt_is_small_and_visible():
    prices = _Prices({"A": _prices(0.5), "B": _prices(0.5)})
    uni = _universe(["A", "B"])
    uni.loc[uni["ticker"] == "B", "institutional_direction_score"] = 1.0
    ranking = screener.rank_universe(uni, 182, load=_loader(prices)).set_index("ticker")
    assert ranking.loc["B", "rank_score"] - ranking.loc["A", "rank_score"] == pytest_approx(screener.INSTITUTIONAL_TILT)
    assert ranking.loc["B", "rank"] == 1


def pytest_approx(x):
    import pytest

    return pytest.approx(x, abs=1e-3)


def test_prefetch_runs_sequentially_once_per_ticker_and_progress_reaches_total():
    prices = _Prices({"A": _prices(0.5), "B": _prices(0.1)})
    seen, progress = [], []
    screener.rank_universe(
        _universe(["A", "B"]), 30, load=_loader(prices), prefetch=seen.append,
        progress=lambda done, total, t: progress.append((done, total)),
    )
    assert seen == ["A", "B"]
    assert progress[-1] == (4, 4)


def test_empty_universe():
    assert screener.rank_universe(pd.DataFrame(), 30).empty


def test_labels():
    assert screener.label_for(None) == "Insufficient data"
    assert screener.label_for(0.5) == "Strong"
    assert screener.label_for(0.15) == "Favorable"
    assert screener.label_for(0.0) == "Neutral"
    assert screener.label_for(-0.2) == "Unfavorable"


def test_ranking_cache_roundtrip_is_keyed_by_horizon_and_universe(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)
    uni_path = tmp_path / "u.parquet"
    pd.DataFrame({"ticker": ["A"]}).to_parquet(uni_path)
    ranking = pd.DataFrame({"ticker": ["A"], "rank": [1]})
    assert screener.load_cached_ranking(3, uni_path) is None
    screener.save_ranking(ranking, 3, uni_path)
    assert screener.load_cached_ranking(3, uni_path)["ticker"].tolist() == ["A"]
    assert screener.load_cached_ranking(182, uni_path) is None  # other horizon -> recompute
