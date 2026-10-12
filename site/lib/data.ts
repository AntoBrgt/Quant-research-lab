// Every number on the site comes from these files, written by src/export_site_data.py.
import verdictsJson from "@/public/data/verdicts.json";
import curvesJson from "@/public/data/equity_curves.json";
import icJson from "@/public/data/ic_by_year.json";
import randomJson from "@/public/data/random_baselines.json";
import phase2Json from "@/public/data/phase2_holdout.json";
import phase1bJson from "@/public/data/phase1b_holdout.json";
import preregJson from "@/public/data/preregistrations.json";
import ledgerJson from "@/public/data/ledger.json";

export const REPO_URL = "https://github.com/AntoBrgt/Quant-research-lab";
export const LOG_URL = `${REPO_URL}/blob/main/docs/RESEARCH_LOG.md`;

export type VerdictRow = {
  id: string;
  phase: string;
  step: string;
  name: string;
  test: string;
  annualized_after_costs: string;
  benchmark_same_period: string;
  verdict: string;
  annualized_pct: number | null;
  benchmark_pct: number | null;
  log_url: string;
};

export type CurvePoint = { date: string; strategy: number; benchmark: number };
export type Curve = { title: string; benchmark: string; note: string; cagr: number; points: CurvePoint[] };

export type Commit = { hash: string; full_hash: string; date: string; url: string };
export type Prereg = {
  id: string;
  title: string;
  source: string;
  frozen_commit: Commit | null;
  protocol_commit?: Commit | null;
  last_commit: Commit | null;
  unchanged_since_freeze: boolean | null;
  markdown: string;
};

export type LedgerArm = {
  name: string;
  picks: number;
  closed: number;
  mean_excess_iwm: number | null;
  ci_low: number | null;
  ci_high: number | null;
  random_pct: number | null;
  verdict?: string;
};
export type LedgerPick = {
  week: string;
  arm: string;
  ticker: string;
  entry_date: string | null;
  status: "open" | "closed";
  exit_date: string | null;
  excess_iwm: number | null;
};
export type Ledger = {
  status: "not_started" | "running" | "closed";
  forward_test_start: string | null;
  arms: LedgerArm[];
  picks: LedgerPick[];
  chain_ok?: boolean;
  generated_at: string;
};

export const meta = { code_commit: verdictsJson.code_commit, generated_at: verdictsJson.generated_at };
export const verdicts = verdictsJson.rows as VerdictRow[];
export const curves = curvesJson.curves as unknown as Record<string, Curve>;
export const icByYear = icJson.step14a;
export const randomBaselines = randomJson.rows;
export const phase2 = phase2Json;
export const phase1b = phase1bJson;
export const preregistrations = preregJson.documents as Prereg[];
export const ledger = ledgerJson as unknown as Ledger;

export const testedRows = verdicts.filter((r) => r.phase !== "3");
export const phase3Row = verdicts.find((r) => r.phase === "3");

export function pct(x: number | null | undefined, digits = 1, signed = true): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "n/a";
  const v = x * 100;
  return `${signed && v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "n/a";
  return x.toFixed(digits);
}

export function shortDate(iso: string): string {
  return iso.slice(0, 10);
}
