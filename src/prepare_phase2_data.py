"""Check (and optionally rebuild) the PHASE 1 data PHASE 2 needs, in order.

    python src/prepare_phase2_data.py            # status only: what exists, what's missing
    python src/prepare_phase2_data.py --build    # build every missing piece (hours on a first run)

`data/` is never committed, so a fresh machine has none of it. Order:
  1. data/processed/backtest/universe_history_2013.parquet  13F history since 2013 (SEC + OpenFIGI)
  2. data/raw/sec_xbrl/companyfacts.zip                    SEC XBRL bulk file (~1.4 GB download)
  3. data/processed/backtest/phase1/step15/members.parquet  python src/run_value_survival.py
  4. data/processed/backtest/phase1/step14a/members.parquet python src/run_smallcap.py
Steps 3-4 run the full PHASE 1 scripts (their models too); everything is cached, a re-run resumes.
Set OPENFIGI_API_KEY (free at openfigi.com) first: without it, CUSIP lookups are paced at 25/minute.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import config
import sec_xbrl
from backtest import data

SRC = Path(__file__).resolve().parent
HISTORY_2013 = data.BACKTEST_DIR / "universe_history_2013.parquet"
STEP15 = data.BACKTEST_DIR / "phase1" / "step15" / "members.parquet"
STEP14A = data.BACKTEST_DIR / "phase1" / "step14a" / "members.parquet"
COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SEC_HEADERS = {"User-Agent": "quant-research-lab research antonin.brengetto@gmail.com"}


def status() -> list[tuple[str, Path, bool]]:
    items = [
        ("13F history since 2013", HISTORY_2013),
        ("SEC XBRL companyfacts.zip", sec_xbrl.COMPANYFACTS_ZIP),
        ("STEP 15 members (top 500)", STEP15),
        ("STEP 14a members (small caps)", STEP14A),
    ]
    return [(name, path, path.exists()) for name, path in items]


def missing_message() -> str:
    lines = ["PHASE 2 needs PHASE 1 data that isn't on this machine:"]
    for name, path, ok in status():
        lines.append(f"  [{'ok' if ok else 'MISSING'}] {name}: {path}")
    lines.append("Build it with:  python src/prepare_phase2_data.py --build")
    return "\n".join(lines)


def build_history() -> None:
    from institutional_research import holdings_13f

    print("1/4 13F history since 2013 (thousands of SEC requests, paced; OpenFIGI for new CUSIPs) ...", flush=True)
    history, summaries = holdings_13f.build_universe_history(
        holdings_13f.load_filers(), 2013, progress=lambda name: print(f"  13F history: {name} ...", flush=True))
    for s in summaries:
        print(f"  {s}", flush=True)
    if history.empty:
        raise RuntimeError("No 13F history built (see the filer statuses above).")
    HISTORY_2013.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(HISTORY_2013, index=False)
    print(f"  saved {len(history):,} rows", flush=True)


def download_companyfacts() -> None:
    import requests

    target = sec_xbrl.COMPANYFACTS_ZIP
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".zip.part")
    print(f"2/4 downloading {COMPANYFACTS_URL} (~1.4 GB) ...", flush=True)
    with requests.get(COMPANYFACTS_URL, headers=SEC_HEADERS, stream=True, timeout=60) as r:
        r.raise_for_status()
        done = 0
        with open(partial, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                print(f"  {done / 1e9:.2f} GB", end="\r", flush=True)
    os.replace(partial, target)
    print(f"\n  saved {target}", flush=True)


def run_script(label: str, script: str) -> None:
    print(f"{label} python src/{script} ...", flush=True)
    subprocess.run([sys.executable, str(SRC / script)], check=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    args = p.parse_args(argv)
    for name, path, ok in status():
        print(f"[{'ok' if ok else 'MISSING'}] {name}: {path}")
    if not args.build:
        return 0 if all(ok for _, _, ok in status()) else 1
    if not os.getenv("OPENFIGI_API_KEY"):
        print("Note: OPENFIGI_API_KEY is not set -- CUSIP lookups will be slow (25/minute).", flush=True)
    if not HISTORY_2013.exists():
        build_history()
    if not sec_xbrl.COMPANYFACTS_ZIP.exists():
        download_companyfacts()
    if not STEP15.exists():
        run_script("3/4", "run_value_survival.py")
    if not STEP14A.exists():
        run_script("4/4", "run_smallcap.py")
    print("All PHASE 2 inputs present. Next: python src/run_phase2.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
