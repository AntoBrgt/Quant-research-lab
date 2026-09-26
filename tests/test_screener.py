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


def _ranking_rows():
    base = dict(company_name="Co", technical_view="POSITIVE", fundamental_view="NEUTRAL", institutional_view="POSITIVE",
                dominant_factors="momentum, trend", confidence="Medium", risk_reward_ratio=2.0)
    return pd.DataFrame([
        base | dict(ticker="A", label="Strong", rank_score=0.5, shortlisted=True, current_price=100.0, stop_loss=95.0, take_profit=110.0),
        base | dict(ticker="B", label="Favorable", rank_score=0.2, shortlisted=True, current_price=50.0, stop_loss=None, take_profit=None),
        base | dict(ticker="C", label="Favorable", rank_score=0.15, shortlisted=False, current_price=10.0, stop_loss=9.0, take_profit=12.0),
        base | dict(ticker="D", label="Unfavorable", rank_score=-0.4, shortlisted=False, current_price=20.0, stop_loss=19.0, take_profit=22.0),
        base | dict(ticker="E", label="Unfavorable", rank_score=-0.2, shortlisted=False, current_price=30.0, stop_loss=None, take_profit=None),
    ])


def test_quick_picks_buys_are_shortlist_sells_are_most_unfavorable():
    buys, sells = screener.quick_picks(_ranking_rows(), n_buy=5, n_sell=5)
    assert buys["ticker"].tolist() == ["A", "B"]  # C is Favorable but not shortlisted
    assert sells["ticker"].tolist() == ["D", "E"]  # worst first
    assert set(buys["action"]) == {"BUY"} and set(sells["action"]) == {"SELL"}


def test_quick_picks_risk_based_sizing_and_cap():
    buys, _ = screener.quick_picks(_ranking_rows(), capital=10_000, risk_pct=0.01)
    a = buys.set_index("ticker").loc["A"]
    # 1% of 10k = 100 risk; 5 per share -> 20 shares = 2,000 (under the 20% cap)
    assert a["shares"] == 20 and a["sizing"] == "risk-based"
    assert a["upside"] == pytest_approx(0.10) and a["downside"] == pytest_approx(-0.05)
    b = buys.set_index("ticker").loc["B"]
    # no stop -> equal split of 10k across 2 picks, capped at 20% -> 2,000 / 50 = 40 shares
    assert b["shares"] == 40 and "equal split" in b["sizing"]

    big_risk, _ = screener.quick_picks(_ranking_rows(), capital=10_000, risk_pct=0.05)
    assert big_risk.set_index("ticker").loc["A", "shares"] == 20  # 100 risk-based shares capped at 20% -> 20
    assert "capped" in big_risk.set_index("ticker").loc["A", "sizing"]


def test_quick_picks_empty():
    buys, sells = screener.quick_picks(pd.DataFrame())
    assert buys.empty and sells.empty


def test_quick_picks_skips_buys_with_poor_risk_reward():
    rows = _ranking_rows()
    rows.loc[rows["ticker"] == "A", "risk_reward_ratio"] = 0.4  # target much closer than stop
    buys, _ = screener.quick_picks(rows)
    assert buys["ticker"].tolist() == ["B"]  # B has no ratio (long-horizon style) -> kept
