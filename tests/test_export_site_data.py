"""Site export: the committed JSON matches the README verdicts, stays small, has no ticker-level picks,
and is deterministic and up to date with the local results (needs_data)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import export_site_data as exp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "public" / "data"
META_KEYS = {"code_commit", "generated_at"}


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _readme_verdict_rows() -> list[dict]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return exp.md_tables_after(text, r"^## Verdicts")[0]


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def test_export_is_small():
    total = sum(p.stat().st_size for p in DATA.glob("*.json"))
    assert 0 < total < exp.MAX_TOTAL_BYTES


def test_no_ticker_level_recommendations():
    forbidden_keys = {"ticker", "tickers", "symbol", "action", "shares", "buy", "sell", "price_target"}
    for path in DATA.glob("*.json"):
        if path.name == "ledger.json":
            continue  # the Phase 3 ledger records past picks by design; checked separately below
        for where, key, value in _walk(json.loads(path.read_text(encoding="utf-8"))):
            assert key.lower() not in forbidden_keys, f"{path.name}{where}"
            if isinstance(value, str):
                assert not re.search(r"\b(BUY|SELL)\b", value), f"{path.name}{where}: {value}"
    ledger = _load("ledger.json")
    assert ledger["status"] in {"not_started", "running", "closed"}
    if ledger["status"] == "not_started":
        assert ledger["picks"] == [] and ledger["forward_test_start"] is None


def test_site_verdicts_match_the_readme_table():
    rows = _load("verdicts.json")["rows"]
    for r in _readme_verdict_rows():
        phase_cell = r["Phase"]
        phase = phase_cell.split()[0]
        steps = re.findall(r"\b(\d+[a-c]?)\b", phase_cell[len(phase):])
        cands = [x for x in rows if x["phase"] == phase and (not steps or x["step"] in steps)]
        if phase == "2":
            h = re.match(r"(H\d)", r["Test"]).group(1)
            cands = [x for x in cands if x["name"].startswith(h)]
        assert cands, f"no site row for README row {r}"
        if phase == "3":
            assert cands[0]["test"] == r["Result"]
            continue
        ann = r["Annualized after costs"]
        if ann != "--":
            assert any(ann in x["annualized_after_costs"] for x in cands), (ann, [x["annualized_after_costs"] for x in cands])
        bench = r["Benchmark, same period"]
        if bench != "--":
            assert any(bench in x["benchmark_same_period"] or x["benchmark_same_period"] in bench for x in cands), bench
        site_verdicts = {x["verdict"].lower() for x in cands}
        for part in r["Verdict"].split(" / "):
            assert any(part.lower() in v for v in site_verdicts), (part, site_verdicts)


def test_phase2_numbers_match_the_readme():
    p2 = {(x["hypothesis"], x["period"]): x for x in _load("phase2_holdout.json")["rows"]}
    for r in _readme_verdict_rows():
        if not r["Phase"].startswith("2"):
            continue
        h = re.match(r"(H\d)", r["Test"]).group(1)
        x = p2[(h, "holdout")]
        assert f"t {x['nw_t']:.2f}" in r["Result"]
        if "random pct" in r["Result"]:
            assert f"random pct {x['random_pct']:.0f}" in r["Result"]
        assert f"{x['portfolio_cagr'] * 100:+.1f}%" in r["Annualized after costs"]
        assert x["verdict"] == "NOT PROVEN" == r["Verdict"]


def test_range_cells_are_not_sortable_numbers():
    assert exp.first_pct("top decile +4.7% to +37.2%") is None
    assert exp.first_pct("+8.1% (2017-26)") == 8.1
    assert exp.first_pct("--") is None


@pytest.mark.needs_data
def test_export_is_deterministic_and_up_to_date(tmp_path):
    first = exp.build()
    second = exp.build()
    exp.write(first, tmp_path / "a")
    exp.write(second, tmp_path / "b")
    for name in first:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
        fresh = {k: v for k, v in json.loads((tmp_path / "a" / name).read_text(encoding="utf-8")).items() if k not in META_KEYS}
        committed = {k: v for k, v in _load(name).items() if k not in META_KEYS}
        assert fresh == committed, f"{name} is stale: run python src/export_site_data.py"
