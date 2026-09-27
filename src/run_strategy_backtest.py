"""STEP 13b -- trade-level backtest of the short-term loop, plus triple-barrier meta-labeling.

    python src/run_strategy_backtest.py                 # weekly PIT panel, all variants, grid, ML, report
    python src/run_strategy_backtest.py --reuse-panel   # skip the panel build (prices/fundamentals)
    python src/run_strategy_backtest.py --seeds 20      # fewer random-entry seeds (faster, coarser percentile)
    python src/run_strategy_backtest.py --reuse-panel --step13c   # STEP 13c: final pre-registered exit test

Point-in-time 13F universe (--universe pit, STEP 11b) on weekly signal dates;
entries at the next day's open; costs 0.10%/side + 0.05% slippage/side.
Writes to data/processed/backtest/strategy/: panel_weekly.parquet,
candidates.parquet, trades_<variant>.parquet, grid.csv, ml_probabilities.parquet, report.md.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import config
from backtest import barrier_model, barriers, data, dataset, strategy

STRATEGY_DIR = data.BACKTEST_DIR / "strategy"
IS_IT_LUCK_PERCENTILE = 95
IS_IT_LUCK_GRID_SHARE = 2 / 3


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


PCT = ("total_return", "cagr", "max_drawdown", "win_rate", "avg_win", "avg_loss", "return", "target_share", "stop_share",
       "base_rate", "precision_top_decile")


def build_weekly_panel(args, history: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    panel = dataset.build_panel(
        None,
        price_loader=lambda t: data.load_long_prices(t, years=args.years),
        fundamentals_loader=None if args.no_fundamentals else data.load_raw_fundamentals,
        benchmark=benchmark,
        horizon_days=strategy.SHORT_HORIZON_DAYS,
        label_days=strategy.MAIN_HOLD,
        rebalance_every=strategy.REBALANCE_EVERY,
        start=args.start,
        progress=lambda i, n, t: print(f"\r[{i}/{n}] {t:<8}", end="", file=sys.stderr, flush=True),
        universe="pit",
        universe_history=history,
    )
    print(file=sys.stderr)
    if not args.no_step13:
        panel = dataset.add_step13_features(panel, data.BACKTEST_DIR / "active_positions.parquet")
    return panel


def run_variant(candidates, result, picks, closes, calendar) -> tuple[strategy.SimResult, dict]:
    sim = strategy.simulate(candidates, result, picks, closes, calendar)
    return sim, strategy.metrics(sim)


def random_distribution(candidates, result, closes, calendar, seeds, mask=None) -> pd.DataFrame:
    rows = []
    for seed in range(seeds):
        sim = strategy.simulate(candidates, result, strategy.random_picks(candidates, seed, mask=mask), closes, calendar)
        m = strategy.metrics(sim)
        rows.append({"seed": seed, "total_return": m.get("total_return"), "expectancy_r": m.get("expectancy_r")})
    return pd.DataFrame(rows)


# STEP 13c: pre-registered, final. Exactly these three exits, nothing else.
STEP13C_HOLD = 20
STEP13C_EXITS = {
    "(a) trailing stop 2 ATR": strategy.Geometry("trail", STEP13C_HOLD, stop_atr=2.0),
    "(b) trailing stop 3 ATR": strategy.Geometry("trail", STEP13C_HOLD, stop_atr=3.0),
    "(c) target 2R / stop 1.5 ATR": strategy.Geometry("grid", STEP13C_HOLD, stop_atr=1.5, target_r=2.0),
}


def run_step13c(candidates, ohlc, closes, calendar, benchmark, seeds: int) -> str:
    """score_rank entries, hold 20, three pre-registered exits; the "Is it luck?" verdict.

    The verdict rule is STEP 13b's. With no grid, its robustness leg is applied to
    the three pre-registered exits: >= 2/3 of them must have positive expectancy.
    Entries: score_rank Strong/Favorable (the Quick Picks risk/reward filter is
    defined on the 60-day-high target these exits replace, so it is not applied).
    """
    picks = strategy.weekly_picks(candidates, strategy.pick_score_rank, geometry_is_app=False)
    rows, verdict_inputs, exposure_rows = [], [], []
    for label, geometry in STEP13C_EXITS.items():
        print(f"STEP 13c {label}: strategy + {seeds} random seeds ...", file=sys.stderr, flush=True)
        result = strategy.outcomes(candidates, ohlc, geometry)
        sim = strategy.simulate(candidates, result, picks, closes, calendar)
        m = strategy.metrics(sim)
        rnd = random_distribution(candidates, result, closes, calendar, seeds)
        lo, hi = strategy.bootstrap_expectancy_ci(sim.trades)
        pct = strategy.percentile_of(m.get("total_return", np.nan), rnd["total_return"])
        if len(sim.trades):
            sim.trades.to_parquet(STRATEGY_DIR / f"trades_13c_{geometry.key}.parquet")
        spy_scaled = strategy.exposure_matched_benchmark(benchmark["adj_close"], sim.exposure)
        rows.append({"exit": label, "trades": m.get("trades"), "expectancy_r": m.get("expectancy_r"),
                     "ci95_low": lo, "ci95_high": hi, "random_percentile": pct,
                     "random_median_return": rnd["total_return"].median(),
                     "total_return": m.get("total_return"), "cagr": m.get("cagr"),
                     "max_drawdown": m.get("max_drawdown"), "sharpe": m.get("sharpe"), "win_rate": m.get("win_rate")})
        exposure_rows.append({"exit": label, "avg_exposure": spy_scaled.get("avg_exposure"),
                              "strategy_return": m.get("total_return"), "strategy_cagr": m.get("cagr"),
                              "spy_at_same_exposure_return": spy_scaled.get("total_return"),
                              "spy_at_same_exposure_cagr": spy_scaled.get("cagr"),
                              "cagr_gap": (m.get("cagr") or np.nan) - (spy_scaled.get("cagr") or np.nan)})
        verdict_inputs.append((label, lo, hi, pct, m.get("expectancy_r")))

    positive = sum(1 for *_, e in verdict_inputs if e is not None and e > 0)
    robust_share = positive / len(verdict_inputs)
    verdict_lines, verdicts, tradeable = [], [], []
    for label, lo, hi, pct, _ in verdict_inputs:
        ok_ci, ok_pct, ok_rob = np.isfinite(lo) and lo > 0, np.isfinite(pct) and pct >= IS_IT_LUCK_PERCENTILE, robust_share >= IS_IT_LUCK_GRID_SHARE
        verdict = "TRADEABLE" if (ok_ci and ok_pct and ok_rob) else "NOT PROVEN"
        verdicts.append(verdict)
        if verdict == "TRADEABLE":
            tradeable.append(label)
        reasons = []
        if not ok_ci:
            reasons.append(f"expectancy CI [{lo:+.3f}, {hi:+.3f}] R includes 0 or is negative")
        if not ok_pct:
            reasons.append(f"beats {pct:.0f}% of random pickers with the same exits (needs >= {IS_IT_LUCK_PERCENTILE})")
        if not ok_rob:
            reasons.append(f"only {positive}/3 pre-registered exits have positive expectancy (needs >= 2/3)")
        verdict_lines.append(f"- **{label}: {verdict}**" + (f" -- {'; '.join(reasons)}." if reasons else "."))
    table = pd.DataFrame(rows).set_index("exit")
    table["verdict"] = verdicts

    final = (f"**Final result: TRADEABLE -- {', '.join(tradeable)}.**" if tradeable else
             "**Final result: NONE of the pre-registered exits is TRADEABLE. The app's short-term loop has no demonstrated edge "
             "after costs; no further strategy variants will be added.**")
    lines = [
        "# STEP 13c -- final, pre-registered exit test",
        "",
        f"- Entries: score_rank (the app's rank_score at {strategy.SHORT_HORIZON_DAYS} days), Strong/Favorable, top {strategy.TOP_K} "
        f"per week, max {strategy.MAX_POSITIONS} positions, {strategy.RISK_PER_TRADE:.0%} equity risk per trade (cap "
        f"{strategy.MAX_POSITION_WEIGHT:.0%}), next-day-open entry, max hold {STEP13C_HOLD} days, costs "
        f"{barriers.COST_PER_SIDE:.2%} + {barriers.SLIPPAGE_PER_SIDE:.2%} per side.",
        f"- Weekly signals {candidates['date'].min():%Y-%m-%d} -> {candidates['date'].max():%Y-%m-%d} "
        f"({candidates['date'].nunique()} weeks), point-in-time 13F universe.",
        "- Exits (pre-registered, no others): (a)/(b) chandelier trailing stop = highest high since entry (completed bars) "
        "- k x ATR(14) at signal, never lowered, no target; (c) stop 1.5 ATR below the signal close, target 2x that risk.",
        f"- Random baseline: {seeds} seeds of random weekly picks from the same universe with the SAME exits, sizing and limits.",
        "- Verdict rule (STEP 13b, unchanged): expectancy 95% CI > 0 (bootstrap by week), >= 95th random percentile, and "
        "robustness >= 2/3 -- with no grid, applied to the three pre-registered exits.",
        "",
        "## Is it luck?",
        _fmt(table, PCT + ("random_median_return",)),
        *verdict_lines,
        "",
        "## Return vs SPY at the strategy's average exposure",
        "SPY held at the strategy's average invested fraction (rest in cash at 0%), same dates. "
        "A negative `cagr_gap` means the loop did worse than simply holding that much SPY.",
        "",
        _fmt(pd.DataFrame(exposure_rows).set_index("exit"),
             ("avg_exposure", "strategy_return", "strategy_cagr", "spy_at_same_exposure_return", "spy_at_same_exposure_cagr", "cagr_gap")),
        final,
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2019-03-01", help="First weekly signal date")
    parser.add_argument("--years", type=int, default=data.DEFAULT_YEARS)
    parser.add_argument("--seeds", type=int, default=strategy.RANDOM_SEEDS)
    parser.add_argument("--no-fundamentals", action="store_true")
    parser.add_argument("--no-step13", action="store_true", help="Skip insider / best-ideas features for the ML filter")
    parser.add_argument("--reuse-panel", action="store_true")
    parser.add_argument("--step13c", action="store_true", help="Only the final pre-registered exit test (report_13c.md)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = STRATEGY_DIR / "panel_weekly.parquet"
    if not data.UNIVERSE_HISTORY_PATH.exists():
        print("No point-in-time universe: run `python src/run_backtest.py --build-universe-history` first.")
        return 1
    history = data.load_universe_history()
    benchmark = data.load_long_prices(config.DEFAULT_BENCHMARK_TICKER, years=args.years).sort_index()
    calendar = benchmark.index

    if args.reuse_panel and panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        panel = build_weekly_panel(args, history, benchmark)
        panel.to_parquet(panel_path)

    prices = {}
    for ticker in panel["ticker"].unique():
        try:
            prices[ticker] = data.load_long_prices(ticker, years=args.years)
        except Exception:
            continue
    print("Candidates + barrier levels ...", file=sys.stderr, flush=True)
    candidates, ohlc = strategy.build_candidates(panel, prices, calendar)
    closes = strategy.forward_filled_closes(ohlc)
    candidates.to_parquet(STRATEGY_DIR / "candidates.parquet")

    if args.step13c:
        report = run_step13c(candidates, ohlc, closes, calendar, benchmark, args.seeds)
        (STRATEGY_DIR / "report_13c.md").write_text(report, encoding="utf-8")
        print(report)
        print(f"Saved: {STRATEGY_DIR / 'report_13c.md'}")
        return 0

    # ---- 1. Main variants: the app's own levels, holds 5/10/20, with random-entry baselines
    main_rows, luck_rows, yearly_sections, results, sims = [], [], [], {}, {}
    randoms = {}
    for hold in strategy.MAX_HOLDS:
        geometry = strategy.Geometry("app", hold)
        results[hold] = strategy.outcomes(candidates, ohlc, geometry)
        print(f"Random baseline, hold {hold} ({args.seeds} seeds) ...", file=sys.stderr, flush=True)
        randoms[hold] = random_distribution(candidates, results[hold], closes, calendar, args.seeds)
        for name, rule in strategy.PICK_RULES.items():
            sim, m = run_variant(candidates, results[hold], strategy.weekly_picks(candidates, rule), closes, calendar)
            sims[(name, hold)] = sim
            m["random_pct_return"] = strategy.percentile_of(m.get("total_return", np.nan), randoms[hold]["total_return"])
            m["random_pct_expectancy"] = strategy.percentile_of(m.get("expectancy_r", np.nan), randoms[hold]["expectancy_r"])
            main_rows.append({"variant": f"{name} / hold {hold}", **m, "skipped_gaps": sim.skipped, "not_opened": sim.not_opened})
            if len(sim.trades):
                sim.trades.to_parquet(STRATEGY_DIR / f"trades_{name}_hold{hold}.parquet")
        rnd = randoms[hold]
        main_rows.append({"variant": f"random (median of {args.seeds}) / hold {hold}",
                          "total_return": rnd["total_return"].median(), "expectancy_r": rnd["expectancy_r"].median()})
    main_table = pd.DataFrame(main_rows).set_index("variant")

    # ---- 2. Benchmarks over the strategy period
    ref = sims[("score_rank", strategy.MAIN_HOLD)].equity
    start, end = ref.index[0], ref.index[-1]
    spy = strategy.buy_and_hold(benchmark["adj_close"], start, end)
    ew = strategy.equal_weight_universe(candidates, closes, calendar)
    ew = ew.loc[start:end]
    ew_metrics = strategy.buy_and_hold(ew, start, end)
    bench_table = pd.DataFrame({f"{config.DEFAULT_BENCHMARK_TICKER} buy & hold": spy, "equal-weight PIT universe (weekly)": ew_metrics}).T

    # ---- 3. Grid: stop ATR x target R x max hold, per signal
    grid_rows = []
    for hold in strategy.MAX_HOLDS:
        for k in strategy.GRID_STOP_ATR:
            for r in strategy.GRID_TARGET_R:
                geometry = strategy.Geometry("grid", hold, k, r)
                res = strategy.outcomes(candidates, ohlc, geometry)
                for name, rule in strategy.PICK_RULES.items():
                    sim = strategy.simulate(candidates, res, strategy.weekly_picks(candidates, rule, geometry_is_app=False), closes, calendar)
                    m = strategy.metrics(sim)
                    grid_rows.append({"signal": name, "stop_atr": k, "target_r": r, "max_hold": hold,
                                      "trades": m.get("trades"), "expectancy_r": m.get("expectancy_r"),
                                      "total_return": m.get("total_return"), "max_drawdown": m.get("max_drawdown"),
                                      "win_rate": m.get("win_rate")})
        print(f"Grid hold {hold} done", file=sys.stderr, flush=True)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(STRATEGY_DIR / "grid.csv", index=False)
    grid_share = grid.assign(ok=grid["expectancy_r"] > 0).groupby("signal")["ok"].mean()

    # ---- 4. Meta-labeling on the main geometry
    main_result = results[strategy.MAIN_HOLD]
    print("Meta-labeling walk-forward ...", file=sys.stderr, flush=True)
    probs = barrier_model.walk_forward_probabilities(candidates, main_result, strategy.MAIN_HOLD, strategy.REBALANCE_EVERY)
    probs.to_parquet(STRATEGY_DIR / "ml_probabilities.parquet")
    oos = probs["p_target"].notna() & (main_result["outcome"] != "skipped")
    y_oos, p_oos = main_result.loc[oos, "y"].astype(int), probs.loc[oos, "p_target"]
    ml_lines = []
    if oos.any():
        top = p_oos >= p_oos.quantile(0.9)
        ml_summary = pd.DataFrame([{
            "oos_trades_labelled": int(oos.sum()),
            "oos_dates": candidates.loc[oos, "date"].nunique(),
            "auc": barrier_model.auc(y_oos.to_numpy(), p_oos.to_numpy()),
            "base_rate": y_oos.mean(),
            "precision_top_decile": y_oos[top].mean(),
        }], index=["meta-label model"])
        calibration = barrier_model.calibration_table(y_oos, p_oos)
        oos_dates = candidates["date"].isin(candidates.loc[probs["p_target"].notna(), "date"].unique())
        taken = oos_dates & (probs["p_target"] >= probs["threshold"])
        filtered_rows = []
        for label, mask in (("score_rank, unfiltered (OOS dates)", oos_dates), ("score_rank + ML filter (P >= train-chosen threshold)", taken)):
            sim = strategy.simulate(candidates, main_result, strategy.weekly_picks(candidates, strategy.pick_score_rank, mask=mask), closes, calendar)
            sims[label] = sim
            filtered_rows.append((label, sim))
        rnd_oos = random_distribution(candidates, main_result, closes, calendar, args.seeds, mask=oos_dates)
        comparison = []
        for label, sim in filtered_rows:
            m = strategy.metrics(sim)
            m["random_pct_return"] = strategy.percentile_of(m.get("total_return", np.nan), rnd_oos["total_return"])
            comparison.append({"variant": label, **m})
        comparison = pd.DataFrame(comparison).set_index("variant")
        thresholds = probs.dropna(subset=["fold"]).groupby("fold")["threshold"].first()
        ml_lines = [
            _fmt(ml_summary, PCT),
            "### Calibration (OOS, deciles of predicted P)",
            _fmt(calibration, PCT),
            f"Thresholds chosen on training data per fold: {', '.join(f'{t:.3f}' for t in thresholds)} (0 = no filter was better in training).",
            "",
            "### Filtered vs unfiltered strategy (same OOS dates, hold 10, app levels)",
            _fmt(comparison[["total_return", "cagr", "max_drawdown", "sharpe", "trades", "win_rate", "expectancy_r",
                             "profit_factor", "random_pct_return"]], PCT),
        ]
        ml_random = rnd_oos
    else:
        ml_lines = ["_Not enough weekly dates for a walk-forward fold._"]
        ml_random = None

    # ---- 5. Is it luck?
    for name in strategy.PICK_RULES:
        sim = sims[(name, strategy.MAIN_HOLD)]
        m = strategy.metrics(sim)
        lo, hi = strategy.bootstrap_expectancy_ci(sim.trades)
        pct = strategy.percentile_of(m.get("total_return", np.nan), randoms[strategy.MAIN_HOLD]["total_return"])
        luck_rows.append((f"{name} / hold {strategy.MAIN_HOLD}", m, lo, hi, pct, float(grid_share.get(name, np.nan))))
    if ml_random is not None:
        label = "score_rank + ML filter (P >= train-chosen threshold)"
        sim = sims[label]
        m = strategy.metrics(sim)
        lo, hi = strategy.bootstrap_expectancy_ci(sim.trades)
        luck_rows.append((label + " / hold 10, OOS dates", m, lo, hi,
                          strategy.percentile_of(m.get("total_return", np.nan), ml_random["total_return"]), np.nan))

    luck_table, verdicts = [], []
    for label, m, lo, hi, pct, share in luck_rows:
        ok_ci = np.isfinite(lo) and lo > 0
        ok_pct = np.isfinite(pct) and pct >= IS_IT_LUCK_PERCENTILE
        ok_grid = np.isfinite(share) and share >= IS_IT_LUCK_GRID_SHARE
        verdict = "TRADEABLE" if (ok_ci and ok_pct and ok_grid) else "NOT PROVEN"
        luck_table.append({"variant": label, "trades": m.get("trades"), "expectancy_r": m.get("expectancy_r"),
                           "ci95_low": lo, "ci95_high": hi, "random_percentile": pct,
                           "grid_cells_profitable": share, "verdict": verdict})
        reasons = []
        if not ok_ci:
            reasons.append(f"expectancy CI [{lo:+.3f}, {hi:+.3f}] R includes 0 or is negative")
        if not ok_pct:
            reasons.append(f"beats only {pct:.0f}% of random pickers (needs >= {IS_IT_LUCK_PERCENTILE})")
        if not ok_grid:
            reasons.append("grid not run for this variant" if not np.isfinite(share) else
                           f"only {share:.0%} of grid cells profitable (needs >= {IS_IT_LUCK_GRID_SHARE:.0%})")
        verdicts.append(f"- **{label}: {verdict}**" + (f" -- {'; '.join(reasons)}." if reasons else
                        " -- expectancy CI above 0 after costs, beats >= 95% of random pickers, robust across the grid."))
    luck_table = pd.DataFrame(luck_table).set_index("variant")

    # ---- Report
    n_members = candidates.groupby("date").size()
    main_cols = ["total_return", "cagr", "max_drawdown", "sharpe", "trades", "win_rate", "avg_win", "avg_loss",
                 "expectancy_r", "profit_factor", "target_share", "stop_share", "random_pct_return", "random_pct_expectancy",
                 "skipped_gaps", "not_opened"]
    grid_view = grid.pivot_table(index=["max_hold", "stop_atr", "target_r"], columns="signal", values="expectancy_r")
    grid_view.index = [f"hold {h} / {k:g} ATR / {r:g}R" for h, k, r in grid_view.index]
    grid_view.index.name = "cell (expectancy R)"
    lines = [
        "# Strategy backtest (STEP 13b) -- the short-term trading loop",
        "",
        f"- Universe: point-in-time 13F (`--universe pit`), weekly signals {candidates['date'].min():%Y-%m-%d} -> "
        f"{candidates['date'].max():%Y-%m-%d} ({candidates['date'].nunique()} weeks, ~{n_members.mean():.0f} priced members/week).",
        f"- Each week: top {strategy.TOP_K} picks, max {strategy.MAX_POSITIONS} open positions, risk {strategy.RISK_PER_TRADE:.0%} "
        f"of equity per trade (cap {strategy.MAX_POSITION_WEIGHT:.0%}), entry next day's open, exit at the first of "
        "stop / target / max hold (same-bar stop+target = stop).",
        f"- Costs: {barriers.COST_PER_SIDE:.2%} + {barriers.SLIPPAGE_PER_SIDE:.2%} slippage per side, on entry and exit.",
        f"- App levels: `risk_reward.compute_risk_reward` at a {strategy.SHORT_HORIZON_DAYS}-day horizon (2.5x ATR stop or tighter "
        "60-day swing low; target = 60-day swing high, else 2R). score_rank = the app's rank_score at that horizon, "
        "with the Quick Picks filter (Strong/Favorable, risk/reward >= 1).",
        f"- Random baseline: {args.seeds} seeds of random weekly picks from the same universe, same exits/sizing/limits. "
        "`random_pct_*` = share of random runs the variant beats.",
        "",
        "## 1. Variants with the app's levels",
        _fmt(main_table[[c for c in main_cols if c in main_table.columns]], PCT),
        "## 2. Benchmarks (same period as score_rank / hold 10)",
        _fmt(bench_table, PCT),
        "## 3. By year (hold 10)",
    ]
    for name in strategy.PICK_RULES:
        lines += [f"**{name}**", "", _fmt(strategy.by_year(sims[(name, strategy.MAIN_HOLD)]), PCT)]
    lines += [
        "## 4. Parameter grid: expectancy (R per trade, after costs) for every cell",
        "Stop = k x ATR(14) below the signal close, target = R x that risk. The Quick Picks risk/reward filter is not applied "
        "in the grid (it's defined on the app's levels).",
        "",
        _fmt(grid_view),
        "Share of cells with positive expectancy: " + ", ".join(f"{s} {v:.0%}" for s, v in grid_share.items()) + ".",
        "",
        "## 5. Meta-labeling: P(target first), out of sample",
        *ml_lines,
        "",
        "## 6. Is it luck?",
        "TRADEABLE only if the 95% expectancy CI (bootstrap by week) is above 0 after costs, the variant beats >= 95% of "
        "random pickers on total return, AND >= 2/3 of its grid cells have positive expectancy. Otherwise NOT PROVEN.",
        "",
        _fmt(luck_table, ("grid_cells_profitable",)),
        *verdicts,
        "",
        "## Caveats",
        "- Universe = 13F-selected names (long US equity, large caps); delisted members without prices can't be traded here.",
        "- Daily bars: intraday order of stop vs target is unknown (stop assumed first); no partial fills, no borrow, no taxes.",
        "- Days to next earnings is not a feature: no point-in-time earnings calendar is available.",
        "- Fundamentals as restated by yfinance; forward P/E unavailable historically.",
    ]
    report = "\n".join(lines) + "\n"
    (STRATEGY_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {STRATEGY_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
