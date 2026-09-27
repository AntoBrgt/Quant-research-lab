"""STEP 11/11b -- backtest the current ranking before any ML model.

    python src/run_backtest.py --build-universe-history     # fetch every 13F since 2016 (SEC + OpenFIGI), then run
    python src/run_backtest.py                              # point-in-time 13F universe if built, else current
    python src/run_backtest.py --universe current           # today's universe on every past date (biased)
    python src/run_backtest.py --tickers AAPL MSFT JPM ...  # a fixed list (current) / a filter (pit)
    python src/run_backtest.py --reuse-panel                # re-evaluate the saved panel, no downloads
    python src/run_backtest.py --reuse-panel --model        # + STEP 12: walk-forward LightGBM / ridge vs baseline
    python src/run_backtest.py --universe pit_broad --model # STEP 13: top-500 PIT universe + insider / best-ideas features

Writes to data/processed/backtest/:
    universe_history.parquet   (filing_date, filer, ticker) 13F selections -- --build-universe-history
    universe_membership.parquet (date, member) PIT universe of this run (pit / pit_broad)
    active_positions.parquet   (STEP 13) active managers' full 13F books per quarter
    panel.parquet              (date, ticker) features + scores + labels -- the ML training set later
    per_date.csv               per-date IC / spread for every signal
    predictions.parquet        (--model) out-of-sample score_ml / score_linear per (date, ticker)
    feature_importance.csv     (--model) mean LightGBM gain across walk-forward folds
    report.md                  the summary printed at the end
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

import config
from backtest import data, dataset, evaluate, model
from institutional_research import holdings_13f

SIGNALS_FULL_HISTORY = ["score_technical", "momentum_12_1", "low_volatility"]
SIGNALS_FUNDAMENTAL_WINDOW = ["score_full", "score_technical", "momentum_12_1", "low_volatility"]
# An equal-weighted universe can legitimately trail or lead a cap-weighted SPY
# for a while, but not by +5%/quarter for nine years (the first, biased run).
UNIVERSE_DRIFT_WARNING = 0.015  # |mean fwd_excess| per label period


def _fmt(frame: pd.DataFrame) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    if frame is None or frame.empty:
        return "_no data_\n"

    def cell(v) -> str:
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            return f"{int(v)}" if (v.is_integer() and abs(v) >= 1) else f"{v:.4f}"
        return "" if v is None else str(v)

    header = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in frame.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [cell(v) for v in row.tolist()]) + " |")
    return "\n".join(lines) + "\n"


def _universe_drift(labelled: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Mean fwd_excess of the whole universe per year -- the survivorship check.

    Averaged per date first (every rebalance date weighs the same), then per year.
    """
    per_date = labelled.groupby("date")["fwd_excess"].mean()
    yearly = per_date.groupby(pd.to_datetime(per_date.index).year).agg(["size", "mean"])
    yearly.index.name = "year"
    yearly.columns = ["dates", "mean_fwd_excess"]
    overall = float(per_date.mean()) if not per_date.empty else float("nan")
    lines = [f"- Mean fwd_excess of the whole universe, all dates: **{overall:+.2%} per label period**."]
    if pd.isna(overall):
        lines.append("- No labelled rows: the drift check could not run.")
    elif abs(overall) > UNIVERSE_DRIFT_WARNING:
        lines.append(
            f"- **WARNING: this is NOT near 0** (|mean| > {UNIVERSE_DRIFT_WARNING:.1%}). The universe as a whole "
            "beats/trails the benchmark by a wide margin, so every signal's bucket returns carry that drift -- "
            "look for remaining selection bias (unpriced delisted members, see coverage) before reading the IC tables. "
            "Part of a gap can be genuine: an equal-weighted universe vs a cap-weighted SPY."
        )
    else:
        lines.append(f"- Near 0 (|mean| <= {UNIVERSE_DRIFT_WARNING:.1%}): no sign of a universe-wide drift.")
    return yearly, lines


def build_report(
    panel: pd.DataFrame, label_days: int, rebalance_every: int, horizon_days: int,
    universe: str = "current", coverage: pd.DataFrame = None, history_summary: str = "",
) -> tuple[str, pd.DataFrame]:
    pit = universe in ("pit", "pit_broad") and "score_rank" in panel.columns
    signals_full = (["score_rank"] if pit else []) + SIGNALS_FULL_HISTORY
    signals_fund = (["score_rank"] if pit else []) + SIGNALS_FUNDAMENTAL_WINDOW
    bucket_score = "score_rank" if pit else "score_full"

    fund_rows = panel[panel["has_fundamentals"]]
    full_table = evaluate.evaluate_signals(panel, signals_full, label_days, rebalance_every)
    fund_table = evaluate.evaluate_signals(fund_rows, signals_fund, label_days, rebalance_every)
    buckets = evaluate.label_buckets(fund_rows if not fund_rows.empty else panel, score=bucket_score)
    per_date = pd.concat(
        [evaluate.per_date_metrics(panel, s).assign(signal=s) for s in dict.fromkeys(signals_full + signals_fund)],
        ignore_index=True,
    )
    yearly = evaluate.by_year(evaluate.per_date_metrics(panel, "score_technical"))
    labelled = panel.dropna(subset=["fwd_excess"])
    drift_table, drift_lines = _universe_drift(labelled)

    if pit and universe == "pit_broad":
        universe_line = (f"- Universe: **point-in-time 13F, broad** -- on each date, the top {dataset.BROAD_TOP_N} US stocks by total "
                         "value across the filers' latest FULL 13F holdings with SEC filing date <= that date (ETFs/funds excluded)"
                         + (f" ({history_summary})" if history_summary else ""))
    elif pit:
        universe_line = ("- Universe: **point-in-time 13F** -- on each date, the tickers in each filer's latest 13F "
                         "with SEC filing date <= that date" + (f" ({history_summary})" if history_summary else ""))
    else:
        universe_line = "- Universe: **current** -- one fixed ticker list on every date (survivorship-biased if it is today's holdings)"

    lines = [
        "# Backtest report -- current ranking vs simple references",
        "",
        universe_line,
        f"- Tickers: {panel['ticker'].nunique()} | rebalance dates: {panel['date'].nunique()} "
        f"({panel['date'].min():%Y-%m-%d} -> {panel['date'].max():%Y-%m-%d}) | labelled rows: {len(labelled)}",
        f"- Scoring horizon: {horizon_days} days ({dataset.horizon.horizon_profile(horizon_days)} profile); "
        f"label: {label_days}-trading-day excess return vs {config.DEFAULT_BENCHMARK_TICKER}, entered next day's close; "
        f"rebalance every {rebalance_every} trading days",
        f"- Rows with point-in-time fundamentals: {len(fund_rows)} "
        f"(from {fund_rows['date'].min():%Y-%m-%d})" if not fund_rows.empty else "- No point-in-time fundamentals available",
        "- NW t-stat: Newey-West t-statistic (overlapping labels). |t| < 2 = indistinguishable from noise.",
    ]
    if pit:
        lines.append("- score_rank = score_full + 0.10 x point-in-time institutional_direction_score: the app's full rank_score.")
    lines.append("")

    if pit:
        lines += [
            "## 0. Universe size and price coverage by year",
            "Share of point-in-time members the backtest could use. `pct_with_ticker`: CUSIP resolved to a ticker; "
            "`pct_priced`: yfinance had a price on the date; `tickers_never_priced`: members with no usable price that "
            "year (delisted, renamed, ticker reused). The unpriced share is invisible to every metric below.",
            "",
            _fmt(coverage),
        ]

    lines += [
        "## 1. Full price history (technicals only" + ("; score_rank uses fundamentals where available)" if pit else ")"),
        _fmt(full_table),
        "## 2. Fundamentals window (same rows for every signal)",
        _fmt(fund_table),
        f"## 3. Do the app's labels mean anything? (fundamentals window, labels cut from {bucket_score})",
        _fmt(buckets),
        "## 4. score_technical rank IC by year",
        _fmt(yearly),
        "## 5. Universe drift: mean fwd_excess of every labelled member, by year",
        _fmt(drift_table),
        *drift_lines,
        "",
        "## Caveats",
    ]
    if pit:
        lines += [
            "- Universe = 13F filings as filed: long US equity positions only; members yfinance can't price "
            "(delisted, renamed, CUSIP unresolved) are counted in section 0 but can't be scored or labelled, "
            "so some survivorship bias remains (worst in the earliest years).",
            "- CUSIP -> ticker uses today's OpenFIGI mapping; a ticker later reused by another company would price the wrong stock.",
        ]
    else:
        lines += [
            "- Universe = today's institutional holdings: survivorship bias (names that shrank or were delisted are missing).",
            "- The 0.10 13F institutional tilt is excluded (run with --universe pit to test it).",
        ]
    lines += [
        "- Fundamentals are as currently restated by yfinance; forward P/E is unavailable historically; "
        "historical market cap is approximated from today's via the price ratio.",
        "- Spreads are gross of trading costs.",
    ]
    return "\n".join(lines) + "\n", per_date


MODEL_SIGNALS = ["score_ml", "score_linear", "score_rank", "score_full", "momentum_12_1", "low_volatility"]
# STEP 13: each new feature is also reported alone, so "the model found nothing" can
# be told apart from "the new information has no signal by itself".
STANDALONE_SIGNALS = model.STEP13_FEATURES + ["institution_count"]


def _score_quintiles(rows: pd.DataFrame, score: str, quantiles: int = 5) -> pd.DataFrame:
    """Mean fwd_excess per per-date quintile of `score` (Q5 = highest)."""
    data = rows.dropna(subset=[score, "fwd_excess"]).copy()
    data["quintile"] = data.groupby("date")[score].transform(
        lambda s: pd.qcut(s.rank(method="first"), quantiles, labels=False) + 1 if len(s) >= quantiles else np.nan
    )
    data = data.dropna(subset=["quintile"])
    grouped = data.groupby(data["quintile"].astype(int))["fwd_excess"].agg(["count", "mean", "median", lambda s: (s > 0).mean()])
    grouped.columns = ["count", "mean_fwd_excess", "median_fwd_excess", "share_beating_benchmark"]
    grouped.index = [f"Q{int(q)}" + (" (top)" if q == quantiles else " (bottom)" if q == 1 else "") for q in grouped.index]
    grouped.index.name = "score_ml quintile"
    return grouped


def build_model_report(
    panel: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame, label_days: int, rebalance_every: int
) -> str:
    """STEP 12 section: every signal on exactly the rows that have an OOS score_ml."""
    rows = panel.merge(predictions[["date", "ticker", "score_ml", "score_linear"]], on=["date", "ticker"], how="left")
    rows = rows[rows["score_ml"].notna()]
    if rows.empty:
        return "\n## 6. STEP 12 -- learned model (out of sample)\n_No out-of-sample predictions (too few dates for the walk-forward)._\n"
    lags = evaluate.overlap_lags(label_days, rebalance_every)
    signals = [s for s in MODEL_SIGNALS if s in rows.columns]
    per_date = {s: evaluate.per_date_metrics(rows, s) for s in signals}
    table = pd.DataFrame({s: evaluate.summarize(per_date[s], lags) for s in signals}).T
    standalone = [s for s in STANDALONE_SIGNALS if s in rows.columns and rows[s].notna().any()]
    standalone_table = pd.DataFrame(
        {s: {**evaluate.summarize(evaluate.per_date_metrics(rows, s), lags), "rows_with_value": int(rows[s].notna().sum())}
         for s in standalone}
    ).T

    yearly = pd.concat(
        {s: evaluate.by_year(per_date[s])["mean_rank_ic"] for s in ("score_ml", "score_rank") if not per_date[s].empty}, axis=1
    )
    yearly.columns = [f"{c}_rank_ic" for c in yearly.columns]
    yearly.index.name = "year"
    top = importance.head(10).set_index("feature") if not importance.empty else importance

    return "\n".join([
        "",
        "## 6. STEP 12/13 -- learned model vs baseline (out of sample only)",
        f"- Walk-forward: expanding window, first {model.MIN_TRAIN_DATES} dates train-only, test blocks of "
        f"{model.TEST_SIZE} dates, embargo {lags + 1} dates. Every signal below is evaluated on the SAME rows: "
        f"the {len(rows)} (date, ticker) rows with an out-of-sample score_ml "
        f"({rows['date'].min():%Y-%m-%d} -> {rows['date'].max():%Y-%m-%d}, {rows['date'].nunique()} dates).",
        "- score_ml: LightGBM on per-date rank features; score_linear: ridge on the same features.",
        "",
        _fmt(table),
        "### New information as standalone signals (same OOS rows)",
        "A date only counts for a signal if the signal varies across names that date (a 0/1 flag that is 0 for everyone "
        "carries no ranking); `dates` shows how many did. `rows_with_value`: rows where the feature is known.",
        "",
        _fmt(standalone_table),
        "### Rank IC by year (OOS)",
        _fmt(yearly),
        "### Top-10 features (mean LightGBM gain across folds)",
        _fmt(top),
        "### Forward excess by score_ml quintile (per date)",
        _fmt(_score_quintiles(rows, "score_ml")),
        "### Promotion rule (README STEP 12)",
        *model.promotion_check(per_date, lags),
        "",
    ])


def _parse_filers(specs) -> dict[str, int]:
    """`Name=CIK` pairs, or names from 13f_filers.csv / the defaults."""
    if not specs:
        return holdings_13f.load_filers()
    known = {**holdings_13f.DEFAULT_FILERS, **holdings_13f.load_filers()}
    filers = {}
    for spec in specs:
        if "=" in spec:
            name, cik = spec.split("=", 1)
            filers[name.strip()] = int(cik)
        elif spec in known:
            filers[spec] = known[spec]
        else:
            raise SystemExit(f"Unknown filer {spec!r}: use Name=CIK or one of {sorted(known)}")
    return filers


def _history_summary(history: pd.DataFrame) -> str:
    if history.empty:
        return "empty"
    return (f"{history['institution'].nunique()} filers: {', '.join(sorted(history['institution'].unique()))}; "
            f"filings {history['filing_date'].min()} -> {history['filing_date'].max()}")


def _add_step13_features(panel: pd.DataFrame, args, out_dir) -> pd.DataFrame:
    return dataset.add_step13_features(
        panel, out_dir / "active_positions.parquet", insider_source=args.insider_source,
        refresh_active=args.refresh_active, max_filing_age_days=args.max_filing_age_days,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", choices=list(dataset.UNIVERSES), default=None,
                        help="Default: the saved panel's universe with --reuse-panel; else pit if universe_history.parquet exists, else current")
    parser.add_argument("--build-universe-history", action="store_true",
                        help="Fetch every 13F-HR since --history-start-year from SEC first (slow on first run)")
    parser.add_argument("--filers", nargs="*", help="For --build-universe-history: Name=CIK pairs or known names (default: 13F filers list)")
    parser.add_argument("--history-start-year", type=int, default=holdings_13f.DEFAULT_HISTORY_START_YEAR)
    parser.add_argument("--max-filing-age-days", type=int, default=dataset.PIT_MAX_FILING_AGE_DAYS)
    parser.add_argument("--broad-top-n", type=int, default=dataset.BROAD_TOP_N, help="pit_broad: members per date")
    parser.add_argument("--tickers", nargs="*", help="current: the ticker list (default: current universe); pit/pit_broad: a filter")
    parser.add_argument("--start", help="First rebalance date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Last rebalance date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=data.DEFAULT_YEARS, help="Years of price history to download")
    parser.add_argument("--horizon-days", type=int, default=dataset.DEFAULT_HORIZON_DAYS)
    parser.add_argument("--label-days", type=int, default=dataset.DEFAULT_LABEL_DAYS)
    parser.add_argument("--rebalance-every", type=int, default=dataset.DEFAULT_REBALANCE_EVERY)
    parser.add_argument("--fundamental-lag-days", type=int, default=dataset.pit_features.DEFAULT_FUNDAMENTAL_LAG_DAYS)
    parser.add_argument("--no-fundamentals", action="store_true", help="Technicals only (no yfinance fundamentals calls)")
    parser.add_argument("--reuse-panel", action="store_true", help="Re-evaluate the saved panel.parquet")
    parser.add_argument("--model", action="store_true", help="STEP 12: walk-forward LightGBM + ridge vs the baseline (pit panels only)")
    parser.add_argument("--no-step13", action="store_true", help="Skip the insider / best-ideas features")
    parser.add_argument("--refresh-step13", action="store_true", help="Recompute the STEP 13 features on a reused panel")
    parser.add_argument("--refresh-active", action="store_true", help="Rebuild the active managers' 13F position history")
    parser.add_argument("--insider-source", choices=["bulk", "per-issuer"], default="bulk",
                        help="Form 4 source: SEC quarterly data sets (fast) or each issuer's Form 4 XMLs (one request per filing)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    out_dir = data.BACKTEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "panel.parquet"
    membership_path = out_dir / "universe_membership.parquet"
    meta_path = out_dir / "panel_meta.json"

    if args.build_universe_history:
        history, summaries = holdings_13f.build_universe_history(
            _parse_filers(args.filers), args.history_start_year,
            progress=lambda name: print(f"13F history: {name} ...", file=sys.stderr, flush=True),
        )
        for s in summaries:
            print(f"  {s}", file=sys.stderr)
        if history.empty:
            print("No 13F history built (see the filer statuses above).")
            return 1
        history.to_parquet(data.UNIVERSE_HISTORY_PATH, index=False)
        print(f"Saved {len(history)} rows to {data.UNIVERSE_HISTORY_PATH}", file=sys.stderr)

    saved_universe = None
    if args.reuse_panel and meta_path.exists():
        saved_universe = json.loads(meta_path.read_text()).get("universe")
    universe = args.universe or saved_universe or ("pit" if data.UNIVERSE_HISTORY_PATH.exists() else "current")
    history = data.load_universe_history() if data.UNIVERSE_HISTORY_PATH.exists() else None
    if universe != "current" and history is None:
        print("No point-in-time universe yet: run with --build-universe-history first (or --universe current).")
        return 1

    if args.reuse_panel:
        panel = pd.read_parquet(panel_path)
        if saved_universe and args.universe and args.universe != saved_universe:
            print(f"Saved panel was built with --universe {saved_universe}; re-run without --reuse-panel for {args.universe}.")
            return 1
        if universe != "current" and "score_rank" not in panel.columns:
            print("Saved panel was built with --universe current; re-run without --reuse-panel for a point-in-time universe.")
            return 1
        membership = pd.read_parquet(membership_path) if universe != "current" and membership_path.exists() else None
    else:
        benchmark = data.load_long_prices(config.DEFAULT_BENCHMARK_TICKER, years=args.years)
        membership = None
        tickers = args.tickers
        if universe == "current":
            tickers = [t.upper() for t in (args.tickers or data.universe_tickers())]
            if not tickers:
                print("No tickers: pass --tickers or build the institutional universe first (see STEP 7).")
                return 1
        else:
            dates = dataset.rebalance_dates(benchmark.sort_index().index, args.rebalance_every, start=args.start, end=args.end)
            if universe == "pit_broad":
                print(f"pit_broad: top {args.broad_top_n} by 13F value on {len(dates)} dates ...", file=sys.stderr, flush=True)
                membership = dataset.pit_broad_universe(history, dates, args.broad_top_n, args.max_filing_age_days)
            else:
                membership = dataset.pit_universe(history, dates, args.max_filing_age_days)
            if args.tickers:  # coverage of the same members the panel was allowed to use
                membership = membership[membership["ticker"].isin({t.upper() for t in args.tickers})]
            membership.to_parquet(membership_path, index=False)
        panel = dataset.build_panel(
            tickers,
            price_loader=lambda t: data.load_long_prices(t, years=args.years),
            fundamentals_loader=None if args.no_fundamentals else data.load_raw_fundamentals,
            benchmark=benchmark,
            horizon_days=args.horizon_days,
            label_days=args.label_days,
            rebalance_every=args.rebalance_every,
            fundamental_lag_days=args.fundamental_lag_days,
            start=args.start,
            end=args.end,
            progress=lambda i, n, t: print(f"\r[{i}/{n}] {t:<8}", end="", file=sys.stderr, flush=True),
            universe=universe,
            universe_history=history,
            max_filing_age_days=args.max_filing_age_days,
            membership=membership,
        )
        print(file=sys.stderr)
        if panel.empty:
            print("Panel is empty -- no ticker had enough price history.")
            return 1
        meta_path.write_text(json.dumps({"universe": universe}))

    step13_missing = any(c not in panel.columns for c in model.INSIDER_FEATURES + model.ACTIVE_FEATURES)
    if universe != "current" and not args.no_step13 and (step13_missing or args.refresh_step13):
        panel = _add_step13_features(panel, args, out_dir)
    panel.to_parquet(panel_path)

    coverage = dataset.universe_coverage(membership, panel) if membership is not None else None
    if coverage is not None:
        coverage.to_csv(out_dir / "universe_coverage.csv")

    report, per_date = build_report(
        panel, args.label_days, args.rebalance_every, args.horizon_days, universe=universe,
        coverage=coverage, history_summary=_history_summary(history) if history is not None and universe != "current" else "",
    )
    if args.model:
        if universe == "current":
            print("--model only runs on a point-in-time panel (--universe pit or pit_broad): the current universe is survivorship-biased.")
            return 1
        model_panel = model.with_institution_count(panel, membership)
        predictions, importance = model.walk_forward_predict(model_panel, args.label_days, args.rebalance_every)
        predictions.to_parquet(out_dir / "predictions.parquet", index=False)
        importance.to_csv(out_dir / "feature_importance.csv", index=False)
        report += build_model_report(model_panel, predictions, importance, args.label_days, args.rebalance_every)
    per_date.to_csv(out_dir / "per_date.csv", index=False)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {panel_path}, {out_dir / 'per_date.csv'}, {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
