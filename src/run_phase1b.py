"""PHASE 1b -- iterate on the development period, promote only what survives selection-bias tests.

    python src/run_phase1b.py                  # dev panel (cached) + every variant, registry, DSR / PBO, promotion
    python src/run_phase1b.py --no-embeddings  # TF-IDF similarity only (embeddings take ~1h on CPU)
    python src/run_phase1b.py --rebuild-panel

Development = 2013-08 -> 2023-12 (protocol.HOLDOUT_START = 2024-01-01). Every
run of every variant is appended to data/processed/backtest/registry.parquet.
The holdout is touched only through protocol.final_test, and only if the
best variant passes DSR > 0.95 and PBO < 0.2. Report: data/processed/backtest/phase1b/report.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

import insider_form4
import sec_xbrl
from backtest import lazy_prices, model, phase1b as p1b, protocol

OUT = protocol.PHASE_DIR


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _fmt(frame: pd.DataFrame, pct=()) -> str:
    if frame is None or frame.empty:
        return "_no data_\n"
    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            if pd.isna(v):
                return ""
            return f"{v:+.1%}" if c in pct else f"{v:.3f}"
        return str(v)
    head = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join([str(i)] + [cell(c, v) for c, v in r.items()]) + " |" for i, r in frame.iterrows()]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Panel (dev by default; the full period only inside final_test)
# ----------------------------------------------------------------------------


def text_features(panel: pd.DataFrame, cik_of: dict, end, use_embeddings: bool) -> pd.DataFrame:
    ciks = sorted({cik_of[t] for t in panel["ticker"].unique() if t in cik_of})
    sections = pd.concat([protocol.filing_sections(c, end) for c in ciks], ignore_index=True)
    sections = sections[sections["status"] == "ok"]
    _log(f"  text: {len(sections)} filings with sections from {sections['cik'].nunique()} issuers")
    ref = sections[pd.to_datetime(sections["filing_date"]) < lazy_prices.IDF_FIT_END]
    ref_texts = [t for col in lazy_prices.SECTIONS for t in ref[col].dropna()]
    vectorizer = lazy_prices.fit_tfidf(ref_texts)
    per_filing = pd.concat([lazy_prices.filing_features(g, vectorizer, use_embeddings) for _, g in sections.groupby("cik")],
                           ignore_index=True)
    return lazy_prices.monthly_features(per_filing, panel, cik_of)


def build_panel(end=None, use_embeddings: bool = True) -> tuple[pd.DataFrame, dict]:
    panel = protocol.monthly_members(p1b.STEP15_MEMBERS, end)
    cik_of = insider_form4.ticker_to_cik()
    _log(f"Panel: {len(panel)} member-months, {panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}")
    _log("  SUE / days since filing ...")
    panel = p1b.add_sue(panel, cik_of, end)
    _log("  Lazy Prices ...")
    panel = panel.join(text_features(panel, cik_of, end, use_embeddings))
    sics = sec_xbrl.sic_codes([cik_of[t] for t in panel["ticker"].unique() if t in cik_of])
    panel["sic2"] = [(sics.get(cik_of.get(t)) or 0) // 100 for t in panel["ticker"]]
    base = [f for f in model.FEATURES if f in panel.columns] + ["sue", "days_since_filing"]
    text = lazy_prices.FEATURES
    numeric = [f for f in base + text if pd.api.types.is_numeric_dtype(panel[f])]
    _log("  industry-relative and 1/3-month changes ...")
    panel = panel.join(p1b.industry_relative(panel, numeric, panel["sic2"]))
    panel = panel.join(p1b.changes(panel, numeric, 1)).join(p1b.changes(panel, numeric, 3))
    _log("  labels (next month) ...")
    panel["next_return"], panel["exit_date"] = p1b.next_month_returns(panel, end)
    panel["fwd_excess"] = panel["next_return"]  # the models rank within dates, so excess vs a benchmark changes nothing
    groups = {
        "base": [f for f in model.FEATURES if f in panel.columns],
        "base+sue": base,
        "base+sue+text": base + text,
        "+industry": base + text + [f"{c}_ind" for c in numeric],
        "+changes": base + text + [f"{c}_ind" for c in numeric] + [f"{c}_chg{m}m" for c in numeric for m in (1, 3)],
    }
    return panel, groups


# ----------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------


def run_variant(panel: pd.DataFrame, feature_set: list[str], kind: str) -> pd.Series:
    """Out-of-sample score per row for one (model kind, feature set)."""
    if kind in ("lgbm_reg", "ridge"):
        preds, _ = model.walk_forward_predict(panel, 21, 21, features=feature_set, min_train=p1b.TRAIN_MONTHS,
                                              test_size=p1b.TEST_MONTHS, embargo=p1b.EMBARGO, pit_checked=True)
        return preds["score_ml" if kind == "lgbm_reg" else "score_linear"].set_axis(panel.index)
    if kind == "lgbm_rank":
        return p1b.walk_forward_rank(panel, feature_set).set_axis(panel.index)
    raise ValueError(kind)


def evaluate_and_register(panel, score: pd.Series, family: str, description: str, params: dict) -> dict:
    rows = panel.assign(_score=score)
    rows = rows[rows["_score"].notna()]
    ic = p1b.evaluate_signal(rows, "_score")
    net = p1b.portfolio(rows, "_score")
    vid = protocol.register(family, description, params, net, ic["mean_rank_ic"], ic["ic_tstat"])
    return {"variant_id": vid, "family": family, "description": description, **{k: v for k, v in ic.items() if k != "per_date"},
            **{f"port_{k}": v for k, v in p1b.perf(net).items()}}


def run_all_variants(panel, groups) -> pd.DataFrame:
    results, scores = [], {}
    for name, feats in groups.items():
        for kind in ("lgbm_reg", "lgbm_rank", "ridge"):
            if kind == "ridge" and ("lgbm_reg", name) in scores:
                continue  # ridge comes with the regressor's walk-forward below
            _log(f"Variant {kind} / {name} ({len(feats)} features) ...")
            if kind == "lgbm_reg":
                preds, _ = model.walk_forward_predict(panel, 21, 21, features=feats, min_train=p1b.TRAIN_MONTHS,
                                                      test_size=p1b.TEST_MONTHS, embargo=p1b.EMBARGO, pit_checked=True)
                scores[("lgbm_reg", name)] = preds["score_ml"].set_axis(panel.index)
                scores[("ridge", name)] = preds["score_linear"].set_axis(panel.index)
            else:
                scores[(kind, name)] = run_variant(panel, feats, kind)
        for kind in ("lgbm_reg", "ridge", "lgbm_rank"):
            results.append(evaluate_and_register(panel, scores[(kind, name)], kind, f"{kind} on {name}",
                                                 {"model": kind, "features": name, "n_features": len(feats)}))
        ens = (p1b.per_date_rank(scores[("lgbm_reg", name)], panel["date"])
               + p1b.per_date_rank(scores[("lgbm_rank", name)], panel["date"])) / 2
        results.append(evaluate_and_register(panel, ens, "ensemble", f"reg+rank ensemble on {name}",
                                             {"model": "ensemble(reg,rank)", "features": name}))
    # single-signal references, same months as the models
    oos = scores[("lgbm_reg", "base")].notna()
    for signal, sign in (("momentum_12_1", 1), ("sue", 1), ("sim_tfidf_rf", 1), ("sim_tfidf_mdna", 1)):
        if signal in panel.columns:
            results.append(evaluate_and_register(panel, (sign * panel[signal]).where(oos), "single_signal",
                                                 f"{signal} alone", {"signal": signal, "sign": sign}))
    return pd.DataFrame(results).set_index("variant_id")


# ----------------------------------------------------------------------------
# Final test (holdout) -- only reached through protocol.final_test
# ----------------------------------------------------------------------------


def make_holdout_runner(variant_row: pd.Series, use_embeddings: bool):
    params = json.loads(variant_row["params"])

    def runner() -> dict:
        panel, groups = build_panel(end=None, use_embeddings=use_embeddings)  # holdout open: full period
        feats = groups[params["features"]]
        if params["model"].startswith("ensemble"):
            s = (p1b.per_date_rank(run_variant(panel, feats, "lgbm_reg"), panel["date"])
                 + p1b.per_date_rank(run_variant(panel, feats, "lgbm_rank"), panel["date"])) / 2
        else:
            s = run_variant(panel, feats, params["model"])
        rows = panel.assign(_score=s)
        rows = rows[rows["_score"].notna() & (rows["date"] >= protocol.HOLDOUT_START)]
        net = p1b.portfolio(rows, "_score")
        cash, spy = _monthly_benchmark("BIL"), _monthly_benchmark("SPY")
        vt = p1b.vol_targeted(net, cash)
        ic = p1b.evaluate_signal(rows, "_score")
        summary = {"strategy": p1b.perf(net), "vol_targeted": p1b.perf(vt), "spy": p1b.perf(spy.reindex(net.index)),
                   "mean_rank_ic": ic["mean_rank_ic"], "ic_tstat": ic["ic_tstat"]}
        return {"summary": summary}

    return runner


def _monthly_benchmark(ticker: str) -> pd.Series:
    """Next-month return per month-end (same convention as the strategy's holding period)."""
    px = protocol.load_prices(ticker)["adj_close"]
    monthly = px.groupby(px.index.to_period("M")).last()
    fwd = monthly.pct_change().shift(-1)
    ends = px.groupby(px.index.to_period("M")).apply(lambda s: s.index.max())
    return pd.Series(fwd.to_numpy(), index=pd.DatetimeIndex(ends.to_numpy())).dropna()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--rebuild-panel", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    use_emb = not args.no_embeddings

    panel_path, groups_path = OUT / "dev_panel.parquet", OUT / "feature_groups.json"
    if panel_path.exists() and groups_path.exists() and not args.rebuild_panel:
        panel, groups = pd.read_parquet(panel_path), json.loads(groups_path.read_text())
        protocol.assert_dev(panel["date"], "cached panel date")
    else:
        panel, groups = build_panel(use_embeddings=use_emb)
        panel.to_parquet(panel_path)
        groups_path.write_text(json.dumps(groups))

    results = run_all_variants(panel, groups)
    results.to_csv(OUT / "variants_this_run.csv")
    reg = protocol.registry()

    # Pre-registered selection: the variant with the best dev Sharpe among THIS run's models.
    models_only = results[results["family"] != "single_signal"]
    candidate = models_only["port_sharpe"].idxmax()
    decision = protocol.promote(candidate)
    _log(f"Candidate {candidate}: DSR {decision['dsr']:.3f}, PBO {decision['pbo']:.3f}, promoted={decision['promoted']}")

    cash = _monthly_benchmark("BIL")
    spy = _monthly_benchmark("SPY")
    cand_returns = protocol.returns_matrix([candidate])[candidate]
    dev_table = pd.DataFrame({
        "candidate (top decile)": p1b.perf(cand_returns),
        "candidate, vol-targeted 15%": p1b.perf(p1b.vol_targeted(cand_returns, cash)),
        "SPY, same months": p1b.perf(spy.reindex(cand_returns.index)),
    }).T

    holdout_lines = ["Not run: the candidate did not meet the promotion rule, so the holdout stays untouched."]
    if decision["promoted"]:
        _log("PROMOTED -- running the one-shot holdout test ...")
        result = protocol.final_test(candidate, make_holdout_runner(reg.set_index("variant_id").loc[candidate], use_emb))
        s = result["summary"]
        holdout = pd.DataFrame({"strategy": s["strategy"], "vol-targeted 15%": s["vol_targeted"], "SPY": s["spy"]}).T
        holdout_lines = [f"Rank IC {s['mean_rank_ic']:.4f} (t {s['ic_tstat']:.2f})", "", _fmt(holdout, ("annualized", "vol", "max_drawdown"))]

    table = results[["description", "mean_rank_ic", "ic_tstat", "port_annualized", "port_sharpe", "port_max_drawdown"]]
    report = "\n".join([
        "# PHASE 1b -- iterate on dev, test once on the holdout",
        "",
        f"- Development {panel['date'].min():%Y-%m} -> {panel['date'].max():%Y-%m}; holdout from {protocol.HOLDOUT_START:%Y-%m-%d} "
        "(never read by any dev loader).",
        f"- Registry: {len(reg)} variants registered in total (all runs, all time); {len(results)} in this run.",
        f"- Walk-forward: {p1b.TRAIN_MONTHS} training months, test blocks of {p1b.TEST_MONTHS}, embargo {p1b.EMBARGO}. "
        "Portfolio: top decile, monthly, 0.15%/side.",
        "",
        "## 1. Variants in this run (dev, out of sample)",
        _fmt(table, ("port_annualized", "port_max_drawdown")),
        "## 2. Selection-bias statistics (over ALL registered variants)",
        f"- Candidate (best dev Sharpe among models, pre-registered rule): `{candidate}` -- {results.loc[candidate, 'description']}",
        f"- Deflated Sharpe ratio: **{decision['dsr']:.3f}** (needs > {protocol.PROMOTION_MIN_DSR}); annual Sharpe "
        f"{decision['sharpe_annual']:.2f} vs the luck benchmark of {decision['n_trials']} trials "
        f"{decision['benchmark_sharpe_per_period'] * np.sqrt(12):.2f}.",
        f"- PBO (CSCV, {protocol.CSCV_BLOCKS} blocks, {decision.get('pbo_combinations', 0)} splits, {decision.get('pbo_variants', 0)} variants): "
        f"**{decision['pbo']:.3f}** (needs < {protocol.PROMOTION_MAX_PBO}); P(in-sample best loses out of sample) "
        f"{decision.get('pbo_p_oos_best_negative', float('nan')):.2f}.",
        f"- **Promoted: {'YES' if decision['promoted'] else 'NO'}**",
        "",
        "### Candidate on dev vs SPY",
        _fmt(dev_table, ("annualized", "vol", "max_drawdown")),
        "## 3. Holdout (one shot)",
        *holdout_lines,
    ])
    (OUT / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
