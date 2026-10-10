"""PHASE 2 check (not part of the verdict): holdout without trades still open at the data end.

    python src/phase2_checks.py      # after run_phase2.py --holdout; reads trades_<H>.parquet

The registered rule counts a trade still open on the last price date with its return so far
(outcome `data_end`). This check drops those trades -- which removes the most recent entries,
so it is a different sample, not a corrected one -- and reports the same statistics.
Delisted stocks also end as `data_end`, but earlier; they stay in.
"""

from __future__ import annotations

import sys

import pandas as pd

import config
from backtest import data, events as ev

OUT_DIR = data.BACKTEST_DIR / "phase2"
HOLDS = {"H1": 63, "H2": 60, "H3": 252}


def main() -> int:
    last = data.load_ext_prices(config.DEFAULT_BENCHMARK_TICKER).index.max()
    lines = ["| Hypothesis | Holdout trades | Open at data end | Mean excess (all) | Mean excess (closed only) | NW t (closed only) | 95% CI (closed only) |",
             "|---|---|---|---|---|---|---|"]
    for key, hold in HOLDS.items():
        trades = ev.period_trades(pd.read_parquet(OUT_DIR / f"trades_{key}.parquet"), "holdout")
        still_open = (trades["outcome"] == "data_end") & (pd.to_datetime(trades["exit_date"]) >= last)
        closed = trades[~still_open]
        lo, hi = ev.bootstrap_ci(closed)
        t = ev.nw_tstat(closed, hold) if len(closed) else float("nan")
        lines.append(f"| {key} | {len(trades)} | {int(still_open.sum())} | {trades['excess'].mean() * 100:+.2f}% | "
                     f"{closed['excess'].mean() * 100:+.2f}% | {t:.2f} | [{lo * 100:+.2f}%, {hi * 100:+.2f}%] |")
    print(f"Last price date {last.date()}\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
