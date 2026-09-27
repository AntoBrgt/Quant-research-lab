"""STEP 11/11b -- backtest the current ranking before any ML model.

    python src/run_backtest.py --build-universe-history     # fetch every 13F since 2016 (SEC + OpenFIGI), then run
    python src/run_backtest.py                              # point-in-time 13F universe if built, else current
    python src/run_backtest.py --universe current           # today's universe on every past date (biased)
    python src/run_backtest.py --tickers AAPL MSFT JPM ...  # a fixed list (current) / a filter (pit)
    python src/run_backtest.py --reuse-panel                # re-evaluate the saved panel, no downloads

Writes to data/processed/backtest/:
    universe_history.parquet   (filing_date, filer, ticker) 13F selections -- --build-universe-history
    universe_membership.parquet (date, member) PIT universe of this run (pit only)
    panel.parquet              (date, ticker) features + scores + labels -- the ML training set later
    per_date.csv               per-date IC / spread for every signal
    report.md                  the summary printed at the end
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import config
from backtest import data, dataset, evaluate
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
    pit = universe == "pit" and "score_rank" in panel.columns
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

    if pit:
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", choices=["current", "pit"], default=None,
                        help="Default: pit if universe_history.parquet exists, else current")
    parser.add_argument("--build-universe-history", action="store_true",
                        help="Fetch every 13F-HR since --history-start-year from SEC first (slow on first run)")
    parser.add_argument("--filers", nargs="*", help="For --build-universe-history: Name=CIK pairs or known names (default: 13F filers list)")
    parser.add_argument("--history-start-year", type=int, default=holdings_13f.DEFAULT_HISTORY_START_YEAR)
    parser.add_argument("--max-filing-age-days", type=int, default=dataset.PIT_MAX_FILING_AGE_DAYS)
    parser.add_argument("--tickers", nargs="*", help="current: the ticker list (default: current universe); pit: a filter")
    parser.add_argument("--start", help="First rebalance date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Last rebalance date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=data.DEFAULT_YEARS, help="Years of price history to download")
    parser.add_argument("--horizon-days", type=int, default=dataset.DEFAULT_HORIZON_DAYS)
    parser.add_argument("--label-days", type=int, default=dataset.DEFAULT_LABEL_DAYS)
    parser.add_argument("--rebalance-every", type=int, default=dataset.DEFAULT_REBALANCE_EVERY)
    parser.add_argument("--fundamental-lag-days", type=int, default=dataset.pit_features.DEFAULT_FUNDAMENTAL_LAG_DAYS)
    parser.add_argument("--no-fundamentals", action="store_true", help="Technicals only (no yfinance fundamentals calls)")
    parser.add_argument("--reuse-panel", action="store_true", help="Re-evaluate the saved panel.parquet")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    out_dir = data.BACKTEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "panel.parquet"
    membership_path = out_dir / "universe_membership.parquet"

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

    universe = args.universe or ("pit" if data.UNIVERSE_HISTORY_PATH.exists() else "current")
    history = None
    if universe == "pit":
        if not data.UNIVERSE_HISTORY_PATH.exists():
            print("No point-in-time universe yet: run with --build-universe-history first (or --universe current).")
            return 1
        history = data.load_universe_history()

    coverage = None
    if args.reuse_panel:
        panel = pd.read_parquet(panel_path)
        if universe == "pit" and "score_rank" not in panel.columns:
            print("Saved panel was built with --universe current; re-run without --reuse-panel for pit.")
            return 1
        membership = pd.read_parquet(membership_path) if universe == "pit" and membership_path.exists() else None
    else:
        benchmark = data.load_long_prices(config.DEFAULT_BENCHMARK_TICKER, years=args.years)
        if universe == "current":
            tickers = [t.upper() for t in (args.tickers or data.universe_tickers())]
            if not tickers:
                print("No tickers: pass --tickers or build the institutional universe first (see STEP 7).")
                return 1
        else:
            tickers = args.tickers
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
        )
        print(file=sys.stderr)
        if panel.empty:
            print("Panel is empty -- no ticker had enough price history.")
            return 1
        panel.to_parquet(panel_path)
        membership = None
        if universe == "pit":
            dates = dataset.rebalance_dates(benchmark.sort_index().index, args.rebalance_every, start=args.start, end=args.end)
            membership = dataset.pit_universe(history, dates, args.max_filing_age_days)
            if args.tickers:  # coverage of the same members the panel was allowed to use
                wanted = {t.upper() for t in args.tickers}
                membership = membership[membership["ticker"].isin(wanted)]
            membership.to_parquet(membership_path, index=False)

    if membership is not None:
        coverage = dataset.universe_coverage(membership, panel)
        coverage.to_csv(out_dir / "universe_coverage.csv")

    report, per_date = build_report(
        panel, args.label_days, args.rebalance_every, args.horizon_days, universe=universe,
        coverage=coverage, history_summary=_history_summary(history) if history is not None else "",
    )
    per_date.to_csv(out_dir / "per_date.csv", index=False)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {panel_path}, {out_dir / 'per_date.csv'}, {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
