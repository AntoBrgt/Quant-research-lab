"""STEP 15 -- "buy cheap quality, sell at target" with survival models (PHASE 1).

    python src/run_value_survival.py            # every stage, cached; re-run resumes
    python src/run_value_survival.py --rebuild  # recompute every stage
    python src/run_value_survival.py --seeds 20 # fewer random portfolios

Needs data/processed/backtest/universe_history_2013.parquet (13F history from
2013, built like STEP 11b with --history-start-year 2013) and the SEC XBRL
bulk file data/raw/sec_xbrl/companyfacts.zip. Writes to
data/processed/backtest/phase1/step15/.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import config
import insider_form4
import sec_xbrl
from backtest import barriers, data, dataset, model, value_survival as vs

OUT_DIR = data.BACKTEST_DIR / "phase1" / "step15"
HISTORY_2013 = data.BACKTEST_DIR / "universe_history_2013.parquet"
MODELS = ("kaplan_meier", "cox", "xgb_aft")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _fmt(frame: pd.DataFrame, pct_cols=()) -> str:
    if frame is None or frame.empty:
        return "_no data_\n"

    def cell(col, v) -> str:
        if isinstance(v, (float, np.floating)):
            if pd.isna(v):
                return ""
            if col in pct_cols:
                return f"{v:+.1%}"
            return f"{int(v)}" if (float(v).is_integer() and abs(v) >= 1) else f"{v:.3f}"
        return "" if v is None else str(v)

    header = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in frame.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [cell(c, v) for c, v in row.items()]) + " |")
    return "\n".join(lines) + "\n"


def _cached(path, build, rebuild: bool):
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def month_ends(calendar: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    cal = calendar[calendar >= pd.Timestamp(start)]
    return list(pd.Series(cal, index=cal).groupby([cal.year, cal.month]).max())


def load_prices(tickers) -> tuple[dict, list]:
    prices, failed = {}, []
    for i, t in enumerate(sorted(tickers), 1):
        if i % 100 == 0:
            _log(f"  prices {i}/{len(tickers)}")
        try:
            prices[t] = data.load_ext_prices(t)
        except Exception:
            failed.append(t)
    return prices, failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2013-08-01")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spy = data.load_ext_prices(config.DEFAULT_BENCHMARK_TICKER).sort_index()
    bil = data.load_ext_prices("BIL").sort_index()
    calendar = spy.index
    dates = month_ends(calendar, args.start)
    history = pd.read_parquet(HISTORY_2013)

    # ---- universe
    _log(f"Universe: pit_broad on {len(dates)} month-ends ...")
    membership = _cached(OUT_DIR / "membership.parquet", lambda: dataset.pit_broad_universe(history, dates), args.rebuild)
    membership["date"] = pd.to_datetime(membership["date"])
    tickers = sorted(membership["ticker"].dropna().unique())
    _log(f"Prices for {len(tickers)} tickers ...")
    prices, unpriced = load_prices(tickers)

    # ---- XBRL fundamentals
    ciks = insider_form4.ticker_to_cik()
    _log("XBRL point-in-time tables ...")
    tables = {t: sec_xbrl.company_table(ciks[t]) for t in prices if t in ciks}
    sics = sec_xbrl.sic_codes([ciks[t] for t in tables])
    raw_closes = {t: data.raw_close(p) for t, p in prices.items()}

    # ---- members + candidates
    def build_members():
        _log("Screen: drawdown / valuation / quality on every member-month ...")
        frame = vs.build_candidates(membership, prices, tables, raw_closes, spy)
        frame = dataset.add_step13_features(frame, OUT_DIR / "active_positions_2012.parquet")
        frame["sector"] = [sec_xbrl.sic_division(sics.get(ciks.get(t))) for t in frame["ticker"]]
        return frame
    members = _cached(OUT_DIR / "members.parquet", build_members, args.rebuild)
    members["date"] = pd.to_datetime(members["date"])
    candidates = members[members["is_candidate"]].reset_index(drop=True)
    _log(f"{len(candidates)} candidate-months, {candidates['ticker'].nunique()} tickers")

    # ---- outcomes
    ohlc = {t: barriers.aligned_ohlc(prices[t], calendar) for t in candidates["ticker"].unique()}
    breaks = {t: sec_xbrl.thesis_break_dates(tables[t]) if t in tables else [] for t in ohlc}

    def build_outcomes():
        _log("Outcomes (targets / thesis break / cap / data end) ...")
        rows = [{"cand": i, "ticker": r.ticker, "date": r.date,
                 **vs.trade_outcome(ohlc[r.ticker], r.date, break_dates=breaks[r.ticker])}
                for i, r in enumerate(candidates[["ticker", "date"]].itertuples(index=False))]
        return pd.DataFrame(rows)
    outcomes = _cached(OUT_DIR / "outcomes.parquet", build_outcomes, args.rebuild)
    ok = outcomes["entry_date"].notna().to_numpy()
    candidates, outcomes = candidates[ok].reset_index(drop=True), outcomes[ok].reset_index(drop=True)
    for c in [c for c in outcomes.columns if c.endswith("_exit_date") or c == "entry_date"]:
        outcomes[c] = pd.to_datetime(outcomes[c])

    # ---- survival models, walk-forward by entry year
    rank_features = [f for f in model.FEATURES if f in candidates.columns]
    x_all = vs.survival_design(candidates, rank_features)
    entry_year = outcomes["entry_date"].dt.year
    years = sorted(entry_year.unique())
    test_years = [y for y in years if y >= years[0] + vs.MIN_TRAIN_YEARS + 1]
    preds, metric_rows, calibrations = {}, [], {}
    grid = np.arange(vs.TRADING_DAYS_PER_MONTH, 36 * vs.TRADING_DAYS_PER_MONTH + 1, vs.TRADING_DAYS_PER_MONTH)
    for target in vs.TARGETS:
        key = f"t{int(round(target * 100))}"
        for name in MODELS:
            parts = []
            oos_d, oos_e, oos_t, oos_p12 = [], [], [], []
            ibs = []
            for year in test_years:
                test_start = pd.Timestamp(year=year, month=1, day=1)
                train_mask = outcomes["entry_date"] < test_start - pd.DateOffset(months=vs.EMBARGO_MONTHS)
                test_mask = entry_year == year
                if train_mask.sum() < 50 or test_mask.sum() == 0:
                    continue
                train = vs.censor_at(outcomes[train_mask], key, test_start, calendar)
                train_idx = train.index
                fitted = vs.fit_predict(name, x_all.loc[train_idx], train[f"{key}_duration"], train[f"{key}_event"],
                                        x_all[test_mask])
                done = train[f"{key}_reason"].isin(["thesis_break", "cap"])
                fail_return = float(train.loc[done, f"{key}_return"].mean()) if done.any() else 0.0
                summary = vs.expected_annualized_return(fitted["surv"], target, fail_return)
                part = pd.DataFrame(summary, index=outcomes.index[test_mask])
                part["fold_year"] = year
                parts.append(part)
                d, e = outcomes.loc[test_mask, f"{key}_duration"], outcomes.loc[test_mask, f"{key}_event"]
                oos_d.append(d); oos_e.append(e)
                oos_t.append(pd.Series(-fitted["risk"], index=d.index))  # risk: higher = sooner hit
                oos_p12.append(part["p_hit_12m"])
                ibs.append(vs.integrated_brier(d, e, fitted["surv"], grid, train[f"{key}_duration"], train[f"{key}_event"]))
            if not parts:
                continue
            pred = pd.concat(parts)
            preds[(name, key)] = pred
            pred.to_parquet(OUT_DIR / f"pred_{name}_{key}.parquet")
            d, e = pd.concat(oos_d), pd.concat(oos_e)
            predicted_time = pd.concat(oos_t)  # higher = later hit
            conc = vs.concordance(d, e, predicted_time) if name != "kaplan_meier" else 0.5
            metric_rows.append({"target": f"+{target:.0%}", "model": name, "oos_rows": len(d), "hit_rate": e.mean(),
                                "concordance": conc, "integrated_brier": float(np.mean(ibs)),
                                "mean_p_hit_12m": pred["p_hit_12m"].mean(),
                                "median_months_median": float(np.median(pred["median_months"][np.isfinite(pred["median_months"])]))
                                if np.isfinite(pred["median_months"]).any() else np.nan})
            calibrations[(name, key)] = vs.calibration_12m(d, e, pd.concat(oos_p12))
    metrics = pd.DataFrame(metric_rows).set_index(["target", "model"])

    # ---- portfolios
    closes = {t: f["close"].ffill().to_numpy(dtype=float) for t, f in ohlc.items()}
    cash_returns = bil["adj_close"].pct_change()
    port_rows, verdict_block = [], []
    random_cache = {}
    for (name, key), pred in preds.items():
        cand = candidates.loc[pred.index, ["date", "ticker"]].assign(score=pred["expected_annualized_return"])
        picks = {d: g.sort_values("score", ascending=False).index[: vs.BUYS_PER_MONTH].tolist() for d, g in cand.groupby("date")}
        start = outcomes.loc[pred.index, "entry_date"].min()
        end = calendar[-1]
        sim = vs.simulate_portfolio(picks, outcomes, key, closes, calendar, cash_returns, start, end)
        if key not in random_cache:
            _log(f"Random portfolios {key} ({args.seeds} seeds) ...")
            rnd = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                rpicks = {d: rng.permutation(g.index.to_numpy())[: vs.BUYS_PER_MONTH].tolist() for d, g in cand.groupby("date")}
                rnd.append(vs.perf(vs.simulate_portfolio(rpicks, outcomes, key, closes, calendar, cash_returns, start, end)["equity"]).get("annualized"))
            random_cache[key] = rnd
        p = vs.perf(sim["equity"])
        verdict, lines, halves = vs.portfolio_verdict(sim["equity"], spy["adj_close"], random_cache[key])
        trades = sim["trades"]
        port_rows.append({"target": key, "model": name, **p, "trades_closed": len(trades), "open_at_end": sim["open_at_end"],
                          "hit_share": (trades["reason"] == "target").mean() if len(trades) else np.nan,
                          "break_share": (trades["reason"] == "thesis_break").mean() if len(trades) else np.nan,
                          "verdict": verdict})
        if name == vs.PRIMARY_MODEL and key == f"t{int(round(vs.PRIMARY_TARGET * 100))}":
            spy_perf = vs.perf(spy["adj_close"].loc[sim["equity"].index[0]:sim["equity"].index[-1]])
            verdict_block = [f"**Primary (pre-registered: {name}, target +{vs.PRIMARY_TARGET:.0%}) -- {verdict}**",
                             f"Period {sim['equity'].index[0]:%Y-%m-%d} -> {sim['equity'].index[-1]:%Y-%m-%d}; SPY annualized "
                             f"{spy_perf['annualized']:+.1%}, Sharpe {spy_perf['sharpe']:.2f}.", "", _fmt(halves, ("strategy_annualized", "spy_annualized")),
                             *lines]
            sim["equity"].to_frame("equity").to_parquet(OUT_DIR / "equity_primary.parquet")
    portfolio = pd.DataFrame(port_rows).set_index(["target", "model"])

    # ---- coverage
    cov_panel = members[["date", "ticker"]].assign(fwd_excess=0.0)
    coverage = dataset.universe_coverage(membership, cov_panel).drop(columns=["pct_labelled"], errors="ignore")
    xbrl_share = members.groupby(members["date"].dt.year)["revenue_ttm"].apply(lambda s: s.notna().mean())
    coverage["pct_priced_with_xbrl_revenue"] = xbrl_share
    cands_by_year = candidates.groupby(candidates["date"].dt.year).agg(candidate_months=("ticker", "size"),
                                                                      tickers=("ticker", "nunique"),
                                                                      price_only_share=("price_only", "mean"))

    report = "\n".join([
        "# STEP 15 -- buy cheap quality, sell at target (survival models)",
        "",
        f"- Universe: pit_broad (top 500 by 13F value, point-in-time), month-ends {dates[0]:%Y-%m} -> {dates[-1]:%Y-%m}.",
        "- Screen: >= 25% below the 52-week high; P/S below its own 5-year median (price-only if < 36 months of P/S "
        "history, flagged); net margin > 0, FCF margin > 0, net debt / market cap < 1 (SEC XBRL, by filing date; missing = excluded).",
        "- Exits: target +20/+30/+50% on the adjusted high; thesis break (2 consecutive quarters with negative FCF and "
        "revenue below a year earlier, known on the 2nd filing date) -> next open; 10-year cap. No price stop. Costs 0.15%/side.",
        f"- Models walk-forward by entry year (>= {vs.MIN_TRAIN_YEARS} training years, 12-month embargo, training labels "
        "censored at the test start). Pre-registered primary: XGBoost AFT at +30%.",
        f"- Portfolio: each month buy up to {vs.BUYS_PER_MONTH} by expected annualized return, max {vs.MAX_OPEN} open, "
        "equal size (1/15 of equity), cash earns BIL. Random baseline: same candidates, random picks, "
        f"{args.seeds} seeds.",
        f"- Tickers never priced by yfinance: {len(unpriced)} of {len(tickers)}.",
        "",
        "## 0. Coverage by year",
        _fmt(coverage, ("pct_with_ticker", "pct_priced", "pct_priced_with_xbrl_revenue")),
        _fmt(cands_by_year, ("price_only_share",)),
        "## 1. Survival models, out of sample",
        "Concordance: 0.5 = no skill (Kaplan-Meier is 0.5 by construction). Integrated Brier: lower is better, "
        "over months 1-36.",
        "",
        _fmt(metrics, ("hit_rate", "mean_p_hit_12m")),
        "### Calibration of P(hit <= 12 months), primary model",
        _fmt(calibrations.get((vs.PRIMARY_MODEL, f"t{int(round(vs.PRIMARY_TARGET * 100))}")), ("predicted", "actual")),
        "## 2. Portfolios (after costs)",
        _fmt(portfolio, ("total_return", "annualized", "max_drawdown", "hit_share", "break_share")),
        "## 3. Verdict (rule fixed before running)",
        "BEATS MARKET only if, after costs: annualized return > SPY over the full period AND each half; beats >= 95% "
        "of random-candidate portfolios; Sharpe >= SPY's.",
        "",
        *verdict_block,
        "",
        "## Caveats",
        "- Fundamentals: SEC XBRL, first-filed values; tag coverage varies by company (see coverage). Delisted names "
        "with no current ticker->CIK mapping have no fundamentals and are excluded by the quality screen.",
        "- Positions open at the end of data are marked to market, never closed with future information.",
        "- Expected annualized return is a ranking heuristic built from the survival curve (see value_survival.py).",
    ])
    (OUT_DIR / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
