"use client";

import { useMemo, useState } from "react";
import type { VerdictRow } from "@/lib/data";
import { VerdictBadge } from "./ui";

type Key = "phase" | "name" | "annualized_pct" | "benchmark_pct" | "verdict";
const COLUMNS: { key: Key; label: string; numeric?: boolean }[] = [
  { key: "phase", label: "Phase · step" },
  { key: "name", label: "Strategy" },
  { key: "annualized_pct", label: "Annualized after costs", numeric: true },
  { key: "benchmark_pct", label: "Benchmark, same period", numeric: true },
  { key: "verdict", label: "Verdict" },
];

const PHASE_ORDER: Record<string, number> = { "1": 1, "1b": 2, "2": 3, "3": 4 };

function compare(a: VerdictRow, b: VerdictRow, key: Key): number {
  if (key === "phase") return (PHASE_ORDER[a.phase] - PHASE_ORDER[b.phase]) || a.id.localeCompare(b.id);
  if (key === "annualized_pct" || key === "benchmark_pct") {
    const x = a[key], y = b[key];
    if (x === null && y === null) return 0;
    if (x === null) return 1; // rows without one number sort last either way
    if (y === null) return -1;
    return x - y;
  }
  return a[key].localeCompare(b[key]);
}

export function VerdictTable({ rows }: { rows: VerdictRow[] }) {
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: "phase", dir: 1 });
  const sorted = useMemo(() => {
    const withNumber = [...rows].sort((a, b) => compare(a, b, sort.key));
    if (sort.dir === 1) return withNumber;
    const numeric = sort.key === "annualized_pct" || sort.key === "benchmark_pct";
    if (!numeric) return withNumber.reverse();
    const has = withNumber.filter((r) => r[sort.key as "annualized_pct"] !== null).reverse();
    return [...has, ...withNumber.filter((r) => r[sort.key as "annualized_pct"] === null)];
  }, [rows, sort]);

  function click(key: Key) {
    setSort((s) => (s.key === key ? { key, dir: s.dir === 1 ? -1 : 1 } : { key, dir: key === "annualized_pct" ? -1 : 1 }));
  }

  return (
    <div className="overflow-x-auto rounded border border-rule bg-surface">
      <table className="w-full text-sm">
        <caption className="sr-only">
          Every tested strategy with its result after costs, the benchmark over the same period, and its verdict. Column
          headers sort the table.
        </caption>
        <thead>
          <tr className="border-b border-rule text-left">
            {COLUMNS.map((c) => (
              <th
                key={c.key}
                scope="col"
                aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
                className={`px-3 py-2 font-medium text-muted whitespace-nowrap ${c.numeric ? "text-right" : ""}`}
              >
                <button type="button" onClick={() => click(c.key)} className="inline-flex items-center gap-1 hover:text-ink">
                  {c.label}
                  <span aria-hidden="true" className="w-3 inline-block">
                    {sort.key === c.key ? (sort.dir === 1 ? "▲" : "▼") : ""}
                  </span>
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.id} className="border-b border-rule last:border-0 align-top">
              <td className="px-3 py-3 whitespace-nowrap text-muted tabular">
                {r.phase}
                {r.step !== r.phase ? ` · ${r.step}` : ""}
              </td>
              <td className="px-3 py-3 min-w-[16rem]">
                <a href={r.log_url} className="text-ink">
                  {r.name}
                </a>
                <div className="text-xs text-muted mt-1">{r.test}</div>
                <div className="sm:hidden mt-2 flex flex-wrap items-center gap-2 text-xs tabular">
                  <VerdictBadge verdict={r.verdict} />
                  {r.annualized_after_costs !== "--" && (
                    <span className="text-ink">
                      {r.annualized_after_costs}
                      {r.benchmark_same_period !== "--" && <span className="text-muted"> vs {r.benchmark_same_period}</span>}
                    </span>
                  )}
                </div>
              </td>
              <td className="px-3 py-3 text-right tabular min-w-[9rem]">{r.annualized_after_costs}</td>
              <td className="px-3 py-3 text-right tabular min-w-[9rem] text-muted">{r.benchmark_same_period}</td>
              <td className="px-3 py-3">
                <VerdictBadge verdict={r.verdict} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
