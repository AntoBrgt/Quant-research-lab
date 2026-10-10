"""PHASE 1b addendum (2026-10-10): the 11 registered variants on the holdout, once.

    python src/run_phase1b_addendum.py           # refuses if the addendum holdout was already run
    python src/run_phase1b_addendum.py --force   # re-run (only after a crash); the report says so

Rules: README "PHASE 1b addendum" and `backtest/phase1b_addendum.py`, committed before this
runs. Builds the panel through the data end with the holdout open (same code as the dev
panel), re-runs each variant's walk-forward over the full period, and judges only signal
months >= 2024-01-01. Writes data/processed/backtest/phase1b/addendum_results.json and
addendum_report.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

import run_phase1b as r1b
from backtest import model, phase1b as p1b, phase1b_addendum as add, protocol

OUT = protocol.PHASE_DIR


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def scores_for_all(panel: pd.DataFrame, groups: dict) -> dict:
    """Out-of-sample score per variant id; lgbm_reg and ridge share one walk-forward, as in the dev run."""
    cache, out = {}, {}
    for vid, (kind, fset) in add.VARIANTS.items():
        feats = groups[fset]
        if kind in ("lgbm_reg", "ridge") and ("reg", fset) not in cache:
            _log(f"  walk-forward lgbm_reg + ridge / {fset} ...")
            preds, _ = model.walk_forward_predict(panel, 21, 21, features=feats, min_train=p1b.TRAIN_MONTHS,
                                                  test_size=p1b.TEST_MONTHS, embargo=p1b.EMBARGO, pit_checked=True)
            cache[("reg", fset)] = preds["score_ml"].set_axis(panel.index)
            cache[("ridge", fset)] = preds["score_linear"].set_axis(panel.index)
        if kind in ("lgbm_rank", "ensemble") and ("rank", fset) not in cache:
            _log(f"  walk-forward lgbm_rank / {fset} ...")
            cache[("rank", fset)] = r1b.run_variant(panel, feats, "lgbm_rank")
        if kind == "ensemble" and ("reg", fset) not in cache:
            preds, _ = model.walk_forward_predict(panel, 21, 21, features=feats, min_train=p1b.TRAIN_MONTHS,
                                                  test_size=p1b.TEST_MONTHS, embargo=p1b.EMBARGO, pit_checked=True)
            cache[("reg", fset)] = preds["score_ml"].set_axis(panel.index)
            cache[("ridge", fset)] = preds["score_linear"].set_axis(panel.index)
        if kind == "lgbm_reg":
            out[vid] = cache[("reg", fset)]
        elif kind == "ridge":
            out[vid] = cache[("ridge", fset)]
        elif kind == "lgbm_rank":
            out[vid] = cache[("rank", fset)]
        else:
            out[vid] = (p1b.per_date_rank(cache[("reg", fset)], panel["date"])
                        + p1b.per_date_rank(cache[("rank", fset)], panel["date"])) / 2
    return out


def runner() -> dict:
    _log("Building the full-period panel (holdout open) ...")
    panel, groups = r1b.build_panel(end=None, use_embeddings=True)
    panel.to_parquet(OUT / "full_panel.parquet")
    scores = scores_for_all(panel, groups)
    spy = r1b._monthly_benchmark("SPY")
    stats, dev_check = {}, {}
    for vid, score in scores.items():
        rows = panel.assign(_score=score)
        rows = rows[rows["_score"].notna() & rows["next_return"].notna()]
        hold = rows[rows["date"] >= protocol.HOLDOUT_START]
        dev = rows[rows["date"] < protocol.HOLDOUT_START]
        ic = add.ic_stats(hold, "_score")
        spread = add.decile_spread(hold, "_score")
        top = p1b.portfolio(hold, "_score")
        stats[vid] = {"model": add.VARIANTS[vid][0], "features": add.VARIANTS[vid][1], "months": ic["months"],
                      "first_month": str(hold["date"].min().date()) if len(hold) else None,
                      "last_month": str(hold["date"].max().date()) if len(hold) else None,
                      "mean_rank_ic": ic["mean_rank_ic"], "ic_t": ic["ic_t"], "pct_months_ic_pos": ic["pct_months_ic_pos"],
                      "spread_mean": float(spread.mean()) if len(spread) else float("nan"),
                      "spread_annualized": add.annualized(spread), "top_decile_annualized": add.annualized(top),
                      "spy_annualized": add.annualized(spy.reindex(top.index))}
        dev_ic = p1b.evaluate_signal(dev, "_score")
        dev_check[vid] = {"dev_mean_rank_ic_now": dev_ic["mean_rank_ic"]}
    verdict = add.verdicts(stats)
    lazy = add.lazy_prices_verdict(stats, verdict)
    summary = {"passed": [v for v, x in verdict.items() if x["passed"]], "lazy_prices": lazy["verdict"]}
    return {"summary": summary, "stats": stats, "verdict": verdict, "lazy_prices": lazy, "dev_check": dev_check}


def render(result: dict) -> str:
    reg = protocol.registry().set_index("variant_id")
    lines = ["# PHASE 1b addendum -- the 11 registered variants on the holdout", "",
             f"Rules fixed {add.ADDENDUM_DATE} before this run (README, PHASE 1b addendum). Holdout signal months "
             f">= {protocol.HOLDOUT_START:%Y-%m-%d}. Holdout looks so far: {result['looks']}"
             + (" -- **FORCED RE-RUN**" if result["looks"] > 1 else ""), "",
             "| Variant | Months | Mean rank IC | NW t | Holm p | IC > 0 months | Decile spread / month (net) | Spread annualized | Top decile annualized | SPY | Checks | Verdict | Dev IC (registry) | Dev IC (this run) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for vid, s in result["stats"].items():
        v = result["verdict"][vid]
        t = s["ic_t"]
        lines.append(
            f"| {s['model']} / {s['features']} | {s['months']} | {s['mean_rank_ic']:+.4f} | {t if t is None else f'{t:.2f}'} | "
            f"{v['holm_p']:.3f} | {s['pct_months_ic_pos']:.0%} | {s['spread_mean'] * 100:+.2f}% | {s['spread_annualized'] * 100:+.1f}% | "
            f"{s['top_decile_annualized'] * 100:+.1f}% | {s['spy_annualized'] * 100:+.1f}% | {sum(v['checks'].values())}/3 | "
            f"{'PASS' if v['passed'] else 'NO EDGE'} | {reg.at[vid, 'mean_rank_ic']:+.4f} | {result['dev_check'][vid]['dev_mean_rank_ic_now']:+.4f} |")
    lz = result["lazy_prices"]
    lines += ["", f"## Lazy Prices: **{lz['verdict']}**", "",
              "| Model | Text variant passes | Holdout IC with text | Without text | Text adds |", "|---|---|---|---|---|"]
    for m, d in lz["by_model"].items():
        lines.append(f"| {m} | {d['text_passes']} | {d['ic_text']:+.4f} | {d['ic_without_text']:+.4f} | {d['text_adds']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    result = protocol.addendum_holdout(add.TEST_ID, runner, force=args.force)
    (OUT / "addendum_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    report = render(result)
    (OUT / "addendum_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
