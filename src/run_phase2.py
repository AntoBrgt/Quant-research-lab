"""PHASE 2 -- can the app's BUY list be trusted? Runs PREREGISTRATION_PHASE2.md, nothing else.

    python src/run_phase2.py                  # development period only (bugs and data checks, no tuning)
    python src/run_phase2.py --holdout        # development + the one-time holdout (needs a clean git tree)
    python src/run_phase2.py --only H1 --seeds 20

Reads the PHASE 1 caches (no rebuild): data/processed/backtest/phase1/step14a/members.parquet,
.../step15/members.parquet, data/raw/prices_ext/, data/raw/form4/bulk/, data/raw/sec_xbrl/.
Writes data/processed/backtest/phase2/: trades_<H>.parquet, results.json, report.md, holdout.lock.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import insider_form4
import sec_xbrl
from backtest import barriers, data, events as ev, phase2_signals as sig
from backtest import value_survival as vs

OUT_DIR = data.BACKTEST_DIR / "phase2"
STEP14A = data.BACKTEST_DIR / "phase1" / "step14a" / "members.parquet"
STEP15 = data.BACKTEST_DIR / "phase1" / "step15" / "members.parquet"
ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILES = ["PREREGISTRATION_PHASE2.md", "src/run_phase2.py", "src/backtest/events.py", "src/backtest/phase2_signals.py"]

HYPOTHESES = {
    "H1": {"name": "Insider cluster buying (small caps)", "bench": "IWM",
           "rule": ev.BarrierExit(hold=63, stop_atr=2.5, cost=0.0030), "hold": 63},
    "H2": {"name": "Post-earnings drift, top-decile SUE (small caps)", "bench": "IWM",
           "rule": ev.BarrierExit(hold=60, stop_atr=None, cost=0.0030), "hold": 60},
    "H3": {"name": "Cheap quality + above 200-day average (top 500)", "bench": "SPY", "rule": None, "hold": 252},
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def load_prices(tickers) -> dict:
    prices = {}
    for t in sorted(set(tickers)):
        try:
            prices[t] = data.load_ext_prices(t)
        except Exception:
            pass
    return prices


def fmt_pct(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.2f}%"


def fmt_num(v, digits=2) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout", action="store_true", help="also run the one-time holdout")
    p.add_argument("--force", action="store_true", help="re-run an already-run holdout (flagged in the report)")
    p.add_argument("--only", nargs="*", choices=list(HYPOTHESES), default=list(HYPOTHESES))
    p.add_argument("--seeds", type=int, default=ev.RANDOM_SEEDS)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import prepare_phase2_data
    if not all(ok for _, _, ok in prepare_phase2_data.status()):
        _log(prepare_phase2_data.missing_message())
        return 1

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = git("status", "--porcelain", "--", *FROZEN_FILES)
    lock = None
    if args.holdout:
        if dirty:
            _log(f"Refusing the holdout: uncommitted changes in the pre-registered files:\n{dirty}")
            return 2
        if args.seeds != ev.RANDOM_SEEDS or set(args.only) != set(HYPOTHESES):
            _log("Refusing the holdout: it runs all hypotheses with the pre-registered 200 seeds.")
            return 2
        lock = ev.holdout_lock(OUT_DIR, commit, args.force)

    spy = data.load_ext_prices(config.DEFAULT_BENCHMARK_TICKER).sort_index()
    iwm = data.load_ext_prices("IWM").sort_index()
    calendar = spy.index
    bench_ohlc = {"SPY": barriers.aligned_ohlc(spy, calendar), "IWM": barriers.aligned_ohlc(iwm, calendar)}
    bench_close = {"SPY": spy["adj_close"].reindex(calendar), "IWM": iwm["adj_close"].reindex(calendar)}
    _log(f"Price calendar {calendar[0].date()} -> {calendar[-1].date()}")

    small = pd.read_parquet(STEP14A)
    small["date"] = pd.to_datetime(small["date"])
    small = small[small["in_universe"].fillna(False).astype(bool)]
    small_elig = ev.Eligibility(small)
    large = pd.read_parquet(STEP15)
    large["date"] = pd.to_datetime(large["date"])
    large_elig = ev.Eligibility(large)
    _log(f"Small-cap universe: {len(small)} member-months, {small['ticker'].nunique()} tickers; "
         f"top-500: {len(large)} member-months")

    ciks = insider_form4.ticker_to_cik()
    results, trades_out, events_count, data_notes = {}, {}, {}, []

    for key in args.only:
        spec = HYPOTHESES[key]
        _log(f"\n== {key}: {spec['name']}")
        if key == "H1":
            tickers = sorted(small["ticker"].unique())
            tx, coverage_end = insider_form4.load_bulk_transactions("2013-07-01", str(calendar[-1].date()))
            data_notes.append(f"H1: Form 4 bulk data covers through {coverage_end}; {len(tx):,} officer/director P/S rows.")
            cik_to_ticker = {}
            for t in tickers:  # only universe tickers; a CIK with several tickers keeps the first alphabetically
                c = ciks.get(t)
                if c is not None and c not in cik_to_ticker:
                    cik_to_ticker[c] = t
            events = sig.insider_cluster_events(tx, cik_to_ticker)
            events = small_elig.filter(events)
            if coverage_end:
                events = events[events["signal_date"] <= pd.Timestamp(coverage_end)]
            elig = small_elig
        elif key == "H2":
            tickers = sorted(small["ticker"].unique())
            sue = {}
            for t in tickers:
                if t in ciks:
                    try:
                        sue[t] = sec_xbrl.sue_series(sec_xbrl.company_table(ciks[t]))
                    except Exception:
                        continue
            data_notes.append(f"H2: SUE history for {len(sue)} of {len(tickers)} small-cap tickers.")
            events = sig.earnings_surprise_events(sue, small_elig.contains)
            elig = small_elig
        else:
            tickers = sorted(large["ticker"].dropna().unique())
            prices_h3 = load_prices(large.loc[large["is_candidate"].fillna(False).astype(bool), "ticker"].unique())
            events = sig.cheap_quality_trend_events(large, prices_h3)
            breaks = {}
            for t in events["ticker"].unique():
                try:
                    breaks[t] = sec_xbrl.thesis_break_dates(sec_xbrl.company_table(ciks[t])) if t in ciks else []
                except Exception:
                    breaks[t] = []
            spec = dict(spec, rule=ev.OutcomeExit(target=0.30, cost=0.0015, cap=252, breaks=breaks))
            elig = large_elig

        events = events.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
        events_count[key] = len(events)
        _log(f"{len(events)} events")
        rule, bench = spec["rule"], spec["bench"]
        needed = set(events["ticker"])
        for pool in elig.pools.values():  # random pickers draw from every month's pool
            needed |= set(pool)
        prices = load_prices(needed)
        ohlc = {t: barriers.aligned_ohlc(px, calendar) for t, px in prices.items()}
        levels = {t: barriers.level_inputs(px) for t, px in prices.items()} if isinstance(rule, ev.BarrierExit) else None
        unpriced = sorted(set(events["ticker"]) - set(prices))
        if unpriced:
            data_notes.append(f"{key}: {len(unpriced)} event tickers have no price history (not traded, counted here).")

        trades = ev.build_trades(events, ohlc, bench_ohlc[bench], rule, levels)
        trades_out[key] = trades
        trades.to_parquet(OUT_DIR / f"trades_{key}.parquet", index=False)
        periods = ["development"] + (["holdout"] if args.holdout else [])
        results[key] = {"name": spec["name"], "benchmark": bench, "events": len(events)}
        for period in periods:
            pt = ev.period_trades(trades, period)
            _log(f"  {period}: {len(pt)} trades; random baseline x{args.seeds} ...")
            rand = ev.random_baseline(pt, elig, ohlc, bench_ohlc[bench], rule, levels, seeds=args.seeds, period=period)
            cost = rule.cost
            sim = ev.simulate_portfolio(pt, ohlc, calendar, cost)
            results[key][period] = ev.evaluate_period(pt, spec["hold"], rand, sim, bench_close[bench])
            r = results[key][period]
            _log(f"  mean excess {fmt_pct(r['mean_excess'])}, t {fmt_num(r['nw_t'])}, "
                 f"random pct {fmt_num(r['random_pct'], 0)}, passed {r['passed']}")
        results[key]["verdict"] = ev.verdict(results[key]["development"], results[key].get("holdout"))

    out = {"commit": commit, "holdout_lock": lock, "notes": data_notes, "results": results}
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    report = render_report(out, args.holdout)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    _log("\n" + report)
    return 0


def render_report(out: dict, holdout: bool) -> str:
    lines = ["# PHASE 2 report", "",
             f"Commit `{out['commit']}`. Rules: PREREGISTRATION_PHASE2.md. "
             + ("Holdout INCLUDED." if holdout else "Development period only -- no verdict yet.")]
    lock = out.get("holdout_lock")
    if lock and len(lock["runs"]) > 1:
        lines.append(f"\n**Warning: the holdout has been run {len(lock['runs'])} times (forced re-run).**")
    lines += ["", "| Hypothesis | Period | Trades | Mean excess / trade | NW t | 95% CI | Random pct | Portfolio CAGR | Bench same exposure | Checks passed |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in out["results"].items():
        for period in ("development", "holdout"):
            if period not in r:
                continue
            x = r[period]
            pm, bm = x.get("portfolio") or {}, x.get("benchmark_same_exposure") or {}
            n_ok = sum(x["checks"].values())
            lines.append(f"| {key} {r['name']} | {period} | {x['trades']} | {fmt_pct(x['mean_excess'])} | {fmt_num(x['nw_t'])} | "
                         f"[{fmt_pct(x['ci_low'])}, {fmt_pct(x['ci_high'])}] | {fmt_num(x['random_pct'], 0)} | "
                         f"{fmt_pct(pm.get('cagr'))} | {fmt_pct(bm.get('cagr'))} ({r['benchmark']}) | {n_ok}/5 |")
    lines += ["", "## Verdicts", ""]
    for key, r in out["results"].items():
        lines.append(f"- **{key}**: {r['verdict']}")
    if out["notes"]:
        lines += ["", "## Data notes", ""] + [f"- {n}" for n in out["notes"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
