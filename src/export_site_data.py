"""Export the research results to small JSON files for the static site (the ONLY bridge to site/).

    python src/export_site_data.py          # -> site/public/data/*.json

Read-only. It reads result files that already exist (the research log's verdict tables,
reports, results.json, saved trades / predictions / equity) and never re-runs a test or
changes a result. Two portfolio curves that were not saved as series are rebuilt from saved
inputs with the same frozen code (Phase 2: `events.simulate_portfolio` on the saved trades;
STEP 14a: `smallcap.monthly_portfolio` on the saved predictions), and the export FAILS if
their CAGR differs from the recorded result.

Deterministic: same inputs + same commit -> byte-identical files. `code_commit` and
`generated_at` are HEAD at export time (the commit the export ran on, so a commit that adds
the JSON points to its parent), never the wall clock. No ticker-level data is exported: curves, tables
and verdicts only.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "site" / "public" / "data"
LOG = ROOT / "docs" / "RESEARCH_LOG.md"
BT = ROOT / "data" / "processed" / "backtest"
SCHEMA_VERSION = 1
LOG_URL = "https://github.com/AntoBrgt/Quant-research-lab/blob/main/docs/RESEARCH_LOG.md"
MAX_TOTAL_BYTES = 2_000_000

# Log section anchors (GitHub slugs of docs/RESEARCH_LOG.md headings)
ANCHORS = {
    "11b": "step-11b----point-in-time-13f-universe-for-the-backtest",
    "12": "step-12----learned-model-vs-the-baseline-out-of-sample-only",
    "13": "step-13----more-power-new-information",
    "13b": "step-13b----trade-level-backtest-of-the-short-term-loop--triple-barrier-meta-labeling",
    "13c": "step-13c----final-pre-registered-exit-test-final-result",
    "15": "step-15----buy-cheap-quality-sell-at-target-survival-models-python-srcrun_value_survivalpy",
    "14a": "step-14a----small-caps-where-institutions-cant-trade-python-srcrun_smallcappy",
    "1b": "phase-1b-addendum----result-holdout-opened-once-2026-10-10-1851-utc-commit-cb5e8eb",
    "2": "phase-2----three-pre-registered-event-rules-python-srcrun_phase2py",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def meta() -> dict:
    return {"schema_version": SCHEMA_VERSION, "code_commit": git("rev-parse", "--short", "HEAD"),
            "generated_at": git("log", "-1", "--format=%cI", "HEAD")}


def rnd(x, digits: int = 6):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return None if not math.isfinite(f) else round(f, digits)


def first_pct(text: str) -> Optional[float]:
    """'+8.1% (2017-26)' -> 8.1 ; '--' or a range ('+4.7% to +37.2%') -> None (not one number to sort by)."""
    if re.search(r"%\s+to\s+[+-]?\d", text or ""):
        return None
    m = re.search(r"([+-]?\d+(?:\.\d+)?)%", text or "")
    return float(m.group(1)) if m else None


def md_tables_after(text: str, heading_regex: str) -> list[list[dict]]:
    """Markdown tables in the section whose heading matches `heading_regex` (until the next heading of any level)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("#") and re.search(heading_regex, l)), None)
    if start is None:
        raise ValueError(f"heading not found: {heading_regex}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    tables, block = [], []
    for line in lines[start + 1:end] + [""]:
        if line.startswith("|"):
            block.append(line)
        elif block:
            tables.append(parse_md_table(block))
            block = []
    return tables


def parse_md_table(block: list[str]) -> list[dict]:
    split = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    header = split(block[0])
    return [dict(zip(header, split(l))) for l in block[2:]]


def monthly_points(equity: pd.Series, bench: pd.Series) -> list[dict]:
    """Month-end points of two cumulative series, both rebased to 1.0 at the first point."""
    frame = pd.DataFrame({"s": equity, "b": bench}).sort_index().ffill().dropna()
    frame = frame.groupby(frame.index.to_period("M")).last()
    frame = frame / frame.iloc[0]
    return [{"date": str(p.end_time.date()), "strategy": rnd(r.s, 4), "benchmark": rnd(r.b, 4)} for p, r in frame.iterrows()]


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


# ----------------------------------------------------------------------------
# verdicts.json: the curated verdict tables of the research log
# ----------------------------------------------------------------------------


def verdict_rows(log_text: str) -> list[dict]:
    rows = []
    for r in md_tables_after(log_text, r"^## PHASE 1 VERDICT")[0]:
        step = r["Step"]
        rows.append({"phase": "1", "step": step, "name": r["Strategy"], "test": r["Test"],
                     "annualized_after_costs": r["Annualized after costs"], "benchmark_same_period": r["SPY / IWM, same period"],
                     "verdict": r["Verdict"]})
    for r in md_tables_after(log_text, r"PHASE 1b addendum -- result")[-1]:
        rows.append({"phase": "1b", "step": "1b", "name": r["Strategy"], "test": r["Test"],
                     "annualized_after_costs": r["Annualized after costs (holdout)"],
                     "benchmark_same_period": r["SPY, same months"], "verdict": r["Verdict"]})
    for r in md_tables_after(log_text, r"^### Holdout \(run once")[-1]:
        rows.append({"phase": "2", "step": "2", "name": r["Strategy"], "test": r["Test"],
                     "annualized_after_costs": r["Annualized after costs (holdout)"],
                     "benchmark_same_period": r["Benchmark, same exposure"], "verdict": r["Verdict"]})
    for i, row in enumerate(rows):
        row["id"] = f"p{row['phase']}-{row['step']}-{i:02d}"
        row["annualized_pct"] = first_pct(row["annualized_after_costs"])
        row["benchmark_pct"] = first_pct(row["benchmark_same_period"])
        row["log_url"] = f"{LOG_URL}#{ANCHORS.get(row['step'], ANCHORS.get(row['phase'], ''))}"
    rows.append({"id": "p3-status", "phase": "3", "step": "17-22", "name": "Own text models + LLM agent + forward-tested pick ledger",
                 "test": phase3_status(), "annualized_after_costs": "--", "benchmark_same_period": "--", "verdict": "NOT STARTED",
                 "annualized_pct": None, "benchmark_pct": None,
                 "log_url": "https://github.com/AntoBrgt/Quant-research-lab/blob/main/README.md"})
    return rows


def phase3_status() -> str:
    pre = ROOT / "PREREGISTRATION_PHASE3.md"
    tracked = bool(git("ls-files", "PREREGISTRATION_PHASE3.md"))
    return "pre-registration committed; forward test not started" if pre.exists() and tracked else \
        "pre-registration drafted; forward test not started"


# ----------------------------------------------------------------------------
# Phase 2: holdout table + equity curves
# ----------------------------------------------------------------------------


def phase2_tables() -> dict:
    res = json.loads((BT / "phase2" / "results.json").read_text(encoding="utf-8"))
    lock = json.loads((BT / "phase2" / "holdout.lock").read_text(encoding="utf-8"))
    out = []
    for key, r in res["results"].items():
        for period in ("development", "holdout"):
            x = r[period]
            out.append({"hypothesis": key, "name": r["name"], "benchmark": r["benchmark"], "period": period,
                        "trades": x["trades"], "mean_excess": rnd(x["mean_excess"]), "nw_t": rnd(x["nw_t"]),
                        "ci_low": rnd(x["ci_low"]), "ci_high": rnd(x["ci_high"]), "random_pct": rnd(x["random_pct"], 1),
                        "portfolio_cagr": rnd((x.get("portfolio") or {}).get("cagr")),
                        "benchmark_same_exposure_cagr": rnd((x.get("benchmark_same_exposure") or {}).get("cagr")),
                        "checks": x["checks"], "checks_passed": int(sum(x["checks"].values()))})
        out[-1]["verdict"] = r["verdict"]
    return {"results_commit": res["commit"], "holdout_runs": lock["runs"], "rows": out,
            "verdicts": {k: r["verdict"] for k, r in res["results"].items()}}


def phase2_curves() -> dict:
    import config
    from backtest import barriers, data, events as ev

    res = json.loads((BT / "phase2" / "results.json").read_text(encoding="utf-8"))
    spy = data.load_ext_prices(config.DEFAULT_BENCHMARK_TICKER).sort_index()
    calendar = spy.index
    bench = {"SPY": spy["adj_close"], "IWM": data.load_ext_prices("IWM").sort_index()["adj_close"]}
    costs = {"H1": 0.0030, "H2": 0.0030, "H3": 0.0015}
    curves = {}
    for key, r in res["results"].items():
        trades = pd.read_parquet(BT / "phase2" / f"trades_{key}.parquet")
        for period in ("development", "holdout"):
            pt = ev.period_trades(trades, period)
            ohlc = {t: barriers.aligned_ohlc(data.load_ext_prices(t), calendar) for t in sorted(pt["ticker"].unique())}
            sim = ev.simulate_portfolio(pt, ohlc, calendar, costs[key])
            equity = sim["equity"]
            recorded = r[period]["portfolio"]["cagr"]
            if abs(cagr(equity) - recorded) > 1e-9:
                raise RuntimeError(f"Phase 2 {key} {period}: rebuilt CAGR {cagr(equity):.6f} != recorded {recorded:.6f}")
            b = bench[r["benchmark"]].reindex(equity.index).ffill()
            curves[f"phase2-{key}-{period}"] = {
                "title": f"{key} {r['name']} -- {period}", "benchmark": f"{r['benchmark']} buy and hold",
                "note": "Strategy: max 20 positions, idle cash at 0%. The verdict compares it with the benchmark at the same average exposure.",
                "cagr": rnd(recorded), "points": monthly_points(equity, b)}
    return curves


# ----------------------------------------------------------------------------
# Phase 1: STEP 15 and STEP 14a curves, STEP 14a IC by year
# ----------------------------------------------------------------------------


def step15_curve() -> dict:
    import config
    from backtest import data

    eq = pd.read_parquet(BT / "phase1" / "step15" / "equity_primary.parquet")["equity"]
    eq.index = pd.to_datetime(eq.index)
    spy = data.load_ext_prices(config.DEFAULT_BENCHMARK_TICKER)["adj_close"].sort_index()
    return {"title": "STEP 15 cheap quality + XGBoost survival (+30% target), primary portfolio", "benchmark": "SPY buy and hold",
            "note": "Cash earns BIL. Same dates as the strategy.", "cagr": rnd(cagr(eq)),
            "points": monthly_points(eq, spy.reindex(eq.index).ffill())}


def step14a_rows():
    import config
    from backtest import data
    import run_smallcap as rs

    panel = pd.read_parquet(BT / "phase1" / "step14a" / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    preds = pd.read_parquet(BT / "phase1" / "step14a" / "predictions.parquet")
    rows = panel.join(preds[["score_ml", "score_linear"]])
    rows = rows[rows["score_ml"].notna()]
    spy = data.load_ext_prices(config.DEFAULT_BENCHMARK_TICKER).sort_index()
    iwm = data.load_ext_prices("IWM").sort_index()
    dates = sorted(panel["date"].unique())
    return rows, rs.benchmark_period_returns(iwm, dates, spy.index), rs.benchmark_period_returns(spy, dates, spy.index)


def step14a_exports() -> tuple[dict, dict]:
    from backtest import evaluate, smallcap as sc

    rows, iwm_r, spy_r = step14a_rows()
    port = sc.monthly_portfolio(rows, "score_ml", rows["next_period_return"])
    iwm_r, spy_r = iwm_r.reindex(port.index), spy_r.reindex(port.index)
    valid = iwm_r.notna() & spy_r.notna()
    net, iwm_r = port["net"][valid], iwm_r[valid]
    ann = sc.annualized(net)
    if abs(ann - 0.064) > 0.0005:  # the log reports +6.4% as priced
        raise RuntimeError(f"STEP 14a: rebuilt annualized {ann:.4f} != recorded +6.4%")
    idx = pd.DatetimeIndex(net.index)
    curve = {"title": "STEP 14a small-cap LightGBM, top decile (as priced)", "benchmark": "IWM, same periods",
             "note": "Monthly rebalance, 0.30%/side. Unpriced members are not included (see the survivorship cases in the log).",
             "cagr": rnd(ann), "points": monthly_points(pd.Series((1 + net.to_numpy()).cumprod(), index=idx),
                                                     pd.Series((1 + iwm_r.to_numpy()).cumprod(), index=idx))}
    series = {}
    for signal in ("score_ml", "momentum_12_1"):
        yearly = evaluate.by_year(evaluate.per_date_metrics(rows, signal))
        series[signal] = [{"year": int(y), "mean_rank_ic": rnd(r["mean_rank_ic"]), "dates": int(r.get("dates", 0))}
                          for y, r in yearly.iterrows()]
    ic = {"title": "STEP 14a small caps: mean rank IC by year (out of sample)", "series": series,
          "note": "Rank IC = per-date Spearman correlation between the score and the next 63-day excess return vs IWM."}
    return curve, ic


# ----------------------------------------------------------------------------
# Random-baseline percentiles (13b, 13c, 15, Phase 2)
# ----------------------------------------------------------------------------


def random_baselines(p2: dict) -> list[dict]:
    out = []
    for r in md_tables_after((BT / "strategy" / "report.md").read_text(encoding="utf-8"), r"^## 6\. Is it luck")[0]:
        out.append({"step": "13b", "name": r["variant"], "percentile": rnd(r["random_percentile"], 1),
                    "baseline": "100 random weekly pickers, same exits and sizing"})
    for r in md_tables_after((BT / "strategy" / "report_13c.md").read_text(encoding="utf-8"), r"^## Is it luck")[0]:
        out.append({"step": "13c", "name": r["exit"], "percentile": rnd(r["random_percentile"], 1),
                    "baseline": "100 random weekly pickers, same exits"})
    m = re.search(r"random portfolios \((\d+)%\)", (BT / "phase1" / "step15" / "report.md").read_text(encoding="utf-8"))
    out.append({"step": "15", "name": "Cheap quality + XGBoost survival (+30%), primary", "percentile": float(m.group(1)),
                "baseline": "100 random-candidate portfolios"})
    for r in p2["rows"]:
        out.append({"step": "2", "name": f"{r['hypothesis']} {r['name']} ({r['period']})", "percentile": r["random_pct"],
                    "baseline": "200 random pickers, same dates and exits"})
    for r in out:
        r["bar"] = 95.0
    return out


# ----------------------------------------------------------------------------
# Phase 1b holdout table
# ----------------------------------------------------------------------------


def phase1b_table() -> dict:
    res = json.loads((BT / "phase1b" / "addendum_results.json").read_text(encoding="utf-8"))
    rows = []
    for vid, s in res["stats"].items():
        v = res["verdict"][vid]
        rows.append({"model": s["model"], "features": s["features"], "months": s["months"],
                     "mean_rank_ic": rnd(s["mean_rank_ic"]), "nw_t": rnd(s["ic_t"]), "holm_p": rnd(v["holm_p"]),
                     "spread_mean_monthly": rnd(s["spread_mean"]), "top_decile_annualized": rnd(s["top_decile_annualized"]),
                     "spy_annualized": rnd(s["spy_annualized"]), "checks": v["checks"],
                     "verdict": "PASS" if v["passed"] else "NO EDGE"})
    return {"holdout_looks": res["looks"], "rows": rows, "lazy_prices": res["lazy_prices"]["verdict"]}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def build() -> dict[str, dict]:
    sys.path.insert(0, str(ROOT / "src"))
    m = meta()
    log_text = LOG.read_text(encoding="utf-8")
    p2 = phase2_tables()
    s14_curve, s14_ic = step14a_exports()
    curves = {"step15-primary": step15_curve(), "step14a-top-decile": s14_curve, **phase2_curves()}
    files = {
        "verdicts.json": {**m, "rows": verdict_rows(log_text)},
        "equity_curves.json": {**m, "curves": curves},
        "ic_by_year.json": {**m, "step14a": s14_ic},
        "random_baselines.json": {**m, "rows": random_baselines(p2)},
        "phase2_holdout.json": {**m, **p2},
        "phase1b_holdout.json": {**m, **phase1b_table()},
        "ledger.json": {**m, "status": "not_started", "forward_test_start": None, "arms": [], "picks": []},
    }
    return files


def write(files: dict[str, dict], out_dir: Path = OUT_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, payload in files.items():
        text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
        total += len(text.encode("utf-8"))
    if total > MAX_TOTAL_BYTES:
        raise RuntimeError(f"export is {total:,} bytes > {MAX_TOTAL_BYTES:,}")
    return total


def main() -> int:
    total = write(build())
    print(f"Wrote {len(list(OUT_DIR.glob('*.json')))} files to {OUT_DIR} ({total / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
