"""STEP 11 -- backtest the current ranking before any ML model.

    python src/run_backtest.py                       # current institutional universe
    python src/run_backtest.py --tickers AAPL MSFT JPM NVDA ...
    python src/run_backtest.py --reuse-panel         # re-evaluate the saved panel, no downloads

Writes to data/processed/backtest/:
    panel.parquet     (date, ticker) features + scores + labels -- the ML training set later
    per_date.csv      per-date IC / spread for every signal
    report.md         the summary printed at the end
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

import config
from backtest import data, dataset, evaluate

SIGNALS_FULL_HISTORY = ["score_technical", "momentum_12_1", "low_volatility"]
SIGNALS_FUNDAMENTAL_WINDOW = ["score_full", "score_technical", "momentum_12_1", "low_volatility"]


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


def build_report(panel: pd.DataFrame, label_days: int, rebalance_every: int, horizon_days: int) -> tuple[str, pd.DataFrame]:
    fund_rows = panel[panel["has_fundamentals"]]
    full_table = evaluate.evaluate_signals(panel, SIGNALS_FULL_HISTORY, label_days, rebalance_every)
    fund_table = evaluate.evaluate_signals(fund_rows, SIGNALS_FUNDAMENTAL_WINDOW, label_days, rebalance_every)
    buckets = evaluate.label_buckets(fund_rows if not fund_rows.empty else panel)
    per_date = pd.concat(
        [evaluate.per_date_metrics(panel, s).assign(signal=s) for s in set(SIGNALS_FULL_HISTORY + SIGNALS_FUNDAMENTAL_WINDOW)],
        ignore_index=True,
    )
    yearly = evaluate.by_year(evaluate.per_date_metrics(panel, "score_technical"))

    labelled = panel.dropna(subset=["fwd_excess"])
    lines = [
        "# Backtest report -- current ranking vs simple references",
        "",
        f"- Tickers: {panel['ticker'].nunique()} | rebalance dates: {panel['date'].nunique()} "
        f"({panel['date'].min():%Y-%m-%d} -> {panel['date'].max():%Y-%m-%d}) | labelled rows: {len(labelled)}",
        f"- Scoring horizon: {horizon_days} days ({dataset.horizon.horizon_profile(horizon_days)} profile); "
        f"label: {label_days}-trading-day excess return vs {config.DEFAULT_BENCHMARK_TICKER}, entered next day's close; "
        f"rebalance every {rebalance_every} trading days",
        f"- Rows with point-in-time fundamentals: {len(fund_rows)} "
        f"(from {fund_rows['date'].min():%Y-%m-%d})" if not fund_rows.empty else "- No point-in-time fundamentals available",
        "- NW t-stat: Newey-West t-statistic (overlapping labels). |t| < 2 = indistinguishable from noise.",
        "",
        "## 1. Full price history (technicals only)",
        _fmt(full_table),
        "## 2. Fundamentals window (same rows for every signal)",
        _fmt(fund_table),
        "## 3. Do the app's labels mean anything? (fundamentals window)",
        _fmt(buckets),
        "## 4. score_technical rank IC by year",
        _fmt(yearly),
        "## Caveats",
        "- Universe = today's institutional holdings: survivorship bias (names that shrank or were delisted are missing).",
        "- Fundamentals are as currently restated by yfinance; forward P/E is unavailable historically; "
        "historical market cap is approximated from today's via the price ratio.",
        "- The 0.10 13F institutional tilt is excluded (no 13F history ingested yet).",
        "- Spreads are gross of trading costs.",
    ]
    return "\n".join(lines) + "\n", per_date


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="*", help="Default: the current institutional universe")
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

    if args.reuse_panel:
        panel = pd.read_parquet(panel_path)
    else:
        tickers = [t.upper() for t in (args.tickers or data.universe_tickers())]
        if not tickers:
            print("No tickers: pass --tickers or build the institutional universe first (see STEP 7).")
            return 1
        benchmark = data.load_long_prices(config.DEFAULT_BENCHMARK_TICKER, years=args.years)
        panel = dataset.build_panel(
            tickers,
            price_loader=lambda t: data.load_long_prices(t, years=args.years),
            fundamentals_loader=None if args.no_fundamentals else data.load_raw_fundamentals,
            benchmark=benchmark,
            horizon_days=args.horizon_days,
            label_days=args.label_days,
            rebalance_every=args.rebalance_every,
            fundamental_lag_days=args.fundamental_lag_days,
            progress=lambda i, n, t: print(f"\r[{i}/{n}] {t:<8}", end="", file=sys.stderr, flush=True),
        )
        print(file=sys.stderr)
        if panel.empty:
            print("Panel is empty -- no ticker had enough price history.")
            return 1
        panel.to_parquet(panel_path)

    report, per_date = build_report(panel, args.label_days, args.rebalance_every, args.horizon_days)
    per_date.to_csv(out_dir / "per_date.csv", index=False)
    (out_dir / "report.md").write_text(report)
    print(report)
    print(f"Saved: {panel_path}, {out_dir / 'per_date.csv'}, {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
