"""STEP 14a -- small caps where institutions can't trade: ML ranking + top-decile portfolio (PHASE 1).

    python src/run_smallcap.py             # every stage, cached; re-run resumes
    python src/run_smallcap.py --rebuild

Needs data/processed/backtest/universe_history_2013.parquet and the SEC XBRL
bulk zip (see run_value_survival.py). Writes to data/processed/backtest/phase1/step14a/.
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import pandas as pd

import config
import insider_form4
import sec_xbrl
import security_master
from backtest import barriers, data, dataset, evaluate, labels, model, pit_features, smallcap as sc, value_survival as vs
from institutional_research import holdings_13f
from run_value_survival import HISTORY_2013, _cached, _fmt, _log, load_prices, month_ends

OUT_DIR = data.BACKTEST_DIR / "phase1" / "step14a"
EXTRA_FEATURES = ["sue", "days_since_filing", "breadth_change_3m"]
BASELINES = ["score_linear", "momentum_12_1", "sue", "insider_buyers_90d", "insider_net_buy_value_90d_mcap"]


def next_open_returns(panel: pd.DataFrame, ohlc: dict, calendar: pd.DatetimeIndex) -> pd.Series:
    """Holding-period return per row: next open after its signal date -> next open after the NEXT signal date.

    A ticker whose prices stop inside the period exits at its last close
    (optimistic for bankruptcies -- the survivorship cases below cover names
    yfinance can't price at all).
    """
    dates = sorted(panel["date"].unique())
    nxt = dict(zip(dates, dates[1:]))
    pos = {d: i for i, d in enumerate(calendar)}
    out = pd.Series(np.nan, index=panel.index)
    for i, (d, t) in enumerate(zip(panel["date"], panel["ticker"])):
        if d not in nxt:
            continue
        a, b = pos[d] + 1, pos[nxt[d]] + 1
        if b >= len(calendar):
            continue
        o = ohlc[t]
        entry = o["open"].iloc[a]
        exit_ = o["open"].iloc[b]
        if pd.isna(exit_):
            closes = o["close"].iloc[a:b + 1].dropna()
            exit_ = closes.iloc[-1] if len(closes) else np.nan
        out.iloc[i] = exit_ / entry - 1 if pd.notna(entry) and entry > 0 and pd.notna(exit_) else np.nan
    return out


def benchmark_period_returns(prices: pd.DataFrame, dates: list, calendar: pd.DatetimeIndex) -> pd.Series:
    pos = {d: i for i, d in enumerate(calendar)}
    o = barriers.aligned_ohlc(prices, calendar)["open"]
    out = {}
    for d0, d1 in zip(dates, dates[1:]):
        a, b = pos[d0] + 1, pos[d1] + 1
        if b < len(calendar):
            out[d0] = o.iloc[b] / o.iloc[a] - 1
    return pd.Series(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2013-08-01")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spy = data.load_ext_prices(config.DEFAULT_BENCHMARK_TICKER).sort_index()
    iwm = data.load_ext_prices("IWM").sort_index()
    calendar = spy.index
    dates = month_ends(calendar, args.start)
    history = pd.read_parquet(HISTORY_2013)

    # ---- full books -> pre-filter -> tickers
    _log("Full 13F books per month-end ...")
    books = _cached(OUT_DIR / "books.parquet", lambda: sc.book_values(history, dates), args.rebuild)
    books["date"] = pd.to_datetime(books["date"])
    band = sc.prefilter(books)
    cusips = sorted(band["cusip"].unique())
    _log(f"Resolving {len(cusips)} CUSIPs (OpenFIGI, cached, paced) ...")
    resolutions = holdings_13f._paced_resolve(security_master.resolve_cusips, cusips, pace=True)
    info = [resolutions.get(c) or {} for c in band["cusip"]]
    band = band.assign(ticker=[(i.get("ticker") or None) for i in info],
                       excluded=[i.get("security_type") in sc.EXCLUDED_TYPES for i in info])
    band = band[~band["excluded"]]
    tickers = sorted(band["ticker"].dropna().unique())
    _log(f"Prices for {len(tickers)} tickers ...")
    prices, failed = load_prices(tickers)
    ciks = insider_form4.ticker_to_cik()
    _log("XBRL tables ...")
    tables = {t: sec_xbrl.company_table(ciks[t]) for t in prices if t in ciks}

    # ---- universe filter + features
    def build_members():
        rows = []
        for ticker, group in band.dropna(subset=["ticker"]).groupby("ticker"):
            if ticker not in prices:
                continue
            px = prices[ticker].sort_index()
            raw = data.raw_close(px)
            dv = sc.dollar_volume_20d(px)
            fund = vs.monthly_fundamentals(tables[ticker], dates) if ticker in tables else pd.DataFrame(index=pd.DatetimeIndex(dates))
            sue = sec_xbrl.sue_series(tables[ticker]) if ticker in tables else pd.DataFrame(columns=["filed", "sue"])
            last_filed = tables[ticker]["filed"].sort_values() if ticker in tables else pd.Series(dtype="datetime64[ns]")
            for r in group.itertuples(index=False):
                d = r.date
                tech = pit_features.pit_technicals(px, d, iwm)
                if not tech:
                    continue
                f = fund.loc[d] if d in fund.index else pd.Series(dtype=float)
                get = lambda k: (None if k not in f or pd.isna(f[k]) else float(f[k]))  # noqa: E731
                price = raw.asof(d)
                shares = get("shares")
                cap = price * shares if shares and pd.notna(price) else None
                rev, ni, fcf = get("revenue_ttm"), get("net_income_ttm"), get("fcf_ttm")
                known_sue = sue[sue["filed"] <= d]
                filed_before = last_filed[last_filed <= d]
                rows.append({
                    "date": d, "ticker": ticker, "cusip": r.cusip, "holders": r.holders,
                    **{k: v for k, v in tech.items() if k != "adj_close"},
                    "raw_price": price, "market_cap": cap, "dollar_volume_20d": dv.asof(d),
                    "log_market_cap": math.log(cap) if cap and cap > 0 else None,
                    "revenue_growth": rev / get("revenue_ttm_prior") - 1 if rev and get("revenue_ttm_prior") else None,
                    "net_margin": ni / rev if rev and ni is not None and rev > 0 else None,
                    "fcf_margin": fcf / rev if rev and fcf is not None and rev > 0 else None,
                    "roe": ni / get("equity") if ni is not None and get("equity") and get("equity") > 0 else None,
                    "fcf_yield": fcf / cap if fcf is not None and cap else None,
                    "net_debt": get("net_debt"),
                    "sue": float(known_sue["sue"].iloc[-1]) if len(known_sue) else None,
                    "days_since_filing": (d - filed_before.iloc[-1]).days if len(filed_before) else None,
                    "institution_count": r.holders,
                })
        frame = pd.DataFrame(rows)
        frame["breadth_change_3m"] = sc.ownership_breadth_change(frame)
        frame["in_universe"] = sc.apply_filters(frame)
        return frame
    members = _cached(OUT_DIR / "members.parquet", build_members, args.rebuild)
    members["date"] = pd.to_datetime(members["date"])
    panel = members[members["in_universe"]].reset_index(drop=True)
    _log(f"Universe: {len(panel)} member-months, {panel['ticker'].nunique()} tickers")

    def add_features():
        frame = dataset.add_step13_features(panel, data.BACKTEST_DIR / "phase1" / "step15" / "active_positions_2012.parquet")
        frame["institutional_direction_score"] = np.nan  # selection-based direction isn't defined on full books
        ohlc = {t: barriers.aligned_ohlc(prices[t], calendar) for t in frame["ticker"].unique()}
        iwm_close = labels.aligned_closes(iwm, calendar)
        fwd = [labels.forward_returns(labels.aligned_closes(prices[t], calendar), iwm_close, d, sc.LABEL_DAYS)
               for d, t in zip(frame["date"], frame["ticker"])]
        frame["fwd_return"] = [a for a, _ in fwd]
        frame["fwd_excess"] = [b for _, b in fwd]
        frame["next_period_return"] = next_open_returns(frame, ohlc, calendar)
        return frame
    panel = _cached(OUT_DIR / "panel.parquet", add_features, args.rebuild)
    panel["date"] = pd.to_datetime(panel["date"])

    # ---- survivorship: unpriced members that the band suggests are small caps
    priced_keys = set(zip(members["date"], members["cusip"]))
    band_small = band[band["rough_mcap"].between(sc.MCAP_MIN, sc.MCAP_MAX)]
    unpriced = band_small[[(d, c) not in priced_keys for d, c in zip(band_small["date"], band_small["cusip"])]]
    unpriced_count = unpriced.groupby("date").size()
    universe_count = panel.groupby("date").size()
    unpriced_share = (unpriced_count / (universe_count + unpriced_count)).fillna(0.0)

    # ---- model + baselines on the same rows
    features = model.FEATURES + EXTRA_FEATURES
    _log("Walk-forward LightGBM ...")
    preds, importance = model.walk_forward_predict(panel, sc.LABEL_DAYS, 21, features=features,
                                                   min_train=36, test_size=6, embargo=sc.EMBARGO, pit_checked=True)
    preds.to_parquet(OUT_DIR / "predictions.parquet")
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    rows = panel.join(preds[["score_ml", "score_linear"]])
    rows = rows[rows["score_ml"].notna()]
    lags = evaluate.overlap_lags(sc.LABEL_DAYS, 21)
    signals = ["score_ml"] + BASELINES
    per_date = {s: evaluate.per_date_metrics(rows, s) for s in signals if s in rows.columns}
    table = pd.DataFrame({s: evaluate.summarize(pd_, lags) for s, pd_ in per_date.items()}).T
    diff_mean, diff_t, diff_n = model.ic_difference_tstat(per_date["score_ml"], per_date["momentum_12_1"], lags)
    yearly = evaluate.by_year(per_date["score_ml"])
    positive_years = int((yearly["mean_rank_ic"] > 0).sum())

    # ---- portfolio: top decile, three survivorship cases, vs IWM and SPY
    port = sc.monthly_portfolio(rows, "score_ml", rows["next_period_return"])
    oos_dates = list(port.index)
    iwm_r = benchmark_period_returns(iwm, sorted(panel["date"].unique()), calendar).reindex(port.index)
    spy_r = benchmark_period_returns(spy, sorted(panel["date"].unique()), calendar).reindex(port.index)
    valid = iwm_r.notna() & spy_r.notna()
    port, iwm_r, spy_r = port[valid], iwm_r[valid], spy_r[valid]
    mid = len(port) // 2
    halves = {"full": slice(None), "first half": slice(0, mid), "second half": slice(mid, None)}
    perf_rows, verdict_ok = [], {}
    for loss in sc.SENSITIVITY_LOSSES:
        net = sc.survivorship_adjusted(port["net"], unpriced_share, loss) if loss > 0 else port["net"]
        label = "as priced" if loss == 0 else f"unpriced members -{loss:.0%} per label period"
        for name, sl in halves.items():
            s, i, p = net.iloc[sl], iwm_r.iloc[sl], spy_r.iloc[sl]
            perf_rows.append({"case": label, "period": name, "months": len(s), "strategy": sc.annualized(s),
                              "iwm": sc.annualized(i), "spy": sc.annualized(p)})
        if loss == 0.50:
            verdict_ok["returns"] = all(r["strategy"] > r["iwm"] and r["strategy"] > r["spy"]
                                        for r in perf_rows if r["case"] == label)
    perf = pd.DataFrame(perf_rows).set_index(["case", "period"])
    checks = [
        ("after costs, -50% survivorship case: annualized > IWM and > SPY, full period and each half", verdict_ok["returns"]),
        (f"rank-IC difference vs momentum NW t > 2 (diff {diff_mean:+.4f}, t = {diff_t:.2f} over {diff_n} dates)"
         if diff_t is not None else "rank-IC difference vs momentum NW t > 2 (n/a)", diff_t is not None and diff_mean > 0 and diff_t > 2),
        (f"IC > 0 in >= 2/3 of OOS years ({positive_years} of {len(yearly)})", positive_years >= math.ceil(len(yearly) * 2 / 3)),
    ]
    verdict = "BEATS MARKET" if all(ok for _, ok in checks) else "DOES NOT BEAT MARKET"

    coverage = pd.DataFrame({
        "universe_members_per_month": universe_count.groupby(universe_count.index.year).mean(),
        "unpriced_small_members_per_month": unpriced_count.groupby(unpriced_count.index.year).mean(),
        "unpriced_share": unpriced_share.groupby(unpriced_share.index.year).mean(),
        "with_xbrl_revenue": panel.groupby(panel["date"].dt.year)["revenue_growth"].apply(lambda s: s.notna().mean()),
        "with_sue": panel.groupby(panel["date"].dt.year)["sue"].apply(lambda s: s.notna().mean()),
    })
    coverage.index.name = "year"
    top = importance.head(12).set_index("feature") if not importance.empty else importance

    report = "\n".join([
        "# STEP 14a -- small caps where institutions can't trade (ML ranking)",
        "",
        f"- Universe (point-in-time, monthly {dates[0]:%Y-%m} -> {dates[-1]:%Y-%m}): US common stocks in the filers' full 13F "
        "books, ETFs/funds/ADRs excluded, market cap $200M-$2B (quoted price x XBRL shares as filed), 20-day average "
        "dollar volume > $1M, price > $3.",
        "- Features: model.FEATURES (per-date ranks) + SUE from XBRL EPS (available from the filing date), days since "
        "the last filing, 13F ownership breadth change (3 months). Target: per-date rank of the 63-day return minus IWM.",
        "- LightGBM walk-forward: 36 training dates, test blocks of 6, embargo 4. Portfolio: monthly, long the top decile "
        "by score_ml, equal weight (max 5%), 0.30%/side on turnover, next-open to next-open.",
        f"- Tickers the pre-filter kept that yfinance could not price: {len(failed)} of {len(tickers)}.",
        "",
        "## 0. Universe and survivorship coverage by year",
        _fmt(coverage, ("unpriced_share", "with_xbrl_revenue", "with_sue")),
        "## 1. Signals on the same OOS rows",
        f"{len(rows)} rows, {rows['date'].nunique()} dates ({rows['date'].min():%Y-%m} -> {rows['date'].max():%Y-%m}).",
        "",
        _fmt(table),
        "### score_ml rank IC by year",
        _fmt(yearly),
        "### Top features",
        _fmt(top),
        "## 2. Top-decile portfolio, annualized, after costs",
        "Survivorship cases: names in the universe yfinance can't price are assumed picked at their universe share and "
        "to lose 50% / 100% over the 63-day label period (smallcap.survivorship_adjusted).",
        "",
        _fmt(perf, ("strategy", "iwm", "spy")),
        "## 3. Verdict (rule fixed before running)",
        *[f"- {'PASS' if ok else 'FAIL'} -- {name}" for name, ok in checks],
        f"- **{verdict}**",
        "",
        "## Caveats",
        "- The -50%/-100% cases are bounds, not estimates; the as-priced case still misses delisted names entirely.",
        "- Rough 13F-implied market cap only pre-filters which names are looked up; membership uses real prices and XBRL shares.",
        "- XBRL coverage for small filers starts in 2011; SUE needs 3 years of quarterly EPS.",
    ])
    (OUT_DIR / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
