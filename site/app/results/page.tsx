import type { Metadata } from "next";
import { EquityChart, ICByYearChart } from "@/components/Charts";
import { PageHeader, Section, VerdictBadge } from "@/components/ui";
import { VerdictTable } from "@/components/VerdictTable";
import { curves, icByYear, num, pct, phase1b, phase2, randomBaselines, verdicts } from "@/lib/data";

export const metadata: Metadata = { title: "Results" };

const CURVE_GROUPS: { title: string; verdictId?: string; keys: string[]; note?: string }[] = [
  { title: "STEP 15 · Cheap quality + survival model", verdictId: "p1-15-10", keys: ["step15-primary"] },
  { title: "STEP 14a · Small-cap LightGBM, top decile", verdictId: "p1-14a-12", keys: ["step14a-top-decile"] },
  { title: "Phase 2 · H1 insider cluster buying", keys: ["phase2-H1-development", "phase2-H1-holdout"] },
  { title: "Phase 2 · H2 top-decile earnings surprise", keys: ["phase2-H2-development", "phase2-H2-holdout"] },
  { title: "Phase 2 · H3 cheap quality + 200-day trend", keys: ["phase2-H3-development", "phase2-H3-holdout"] },
];

function verdictFor(group: (typeof CURVE_GROUPS)[number]): string {
  if (group.verdictId) return verdicts.find((v) => v.id === group.verdictId)?.verdict ?? "";
  const h = group.keys[0].split("-")[1] as keyof typeof phase2.verdicts;
  return phase2.verdicts[h];
}

export default function Results() {
  const ic = icByYear.series.score_ml.map((p, i) => ({
    year: p.year,
    model: p.mean_rank_ic,
    momentum: icByYear.series.momentum_12_1[i]?.year === p.year ? icByYear.series.momentum_12_1[i].mean_rank_ic : null,
  }));
  const positiveYears = ic.filter((p) => (p.model ?? 0) > 0).length;

  return (
    <div>
      <PageHeader eyebrow="Results" title="Every test and its verdict">
        Annualized returns are after costs, over each strategy&rsquo;s own test period. Click a column header to sort.
        Each name links to the step in the research log.
      </PageHeader>

      <Section id="verdicts" title="Verdicts">
        <VerdictTable rows={verdicts} />
      </Section>

      <Section id="curves" title="Growth of $1 vs the benchmark">
        <p className="text-muted mb-6 max-w-3xl">
          Monthly points, rebased to 1 at each test&rsquo;s first month. A curve above its benchmark is not a pass by
          itself: the verdict also needs statistical significance, a win against random pickers, and (for partly
          invested strategies) a win against the benchmark held at the same exposure.
        </p>
        <div className="space-y-10">
          {CURVE_GROUPS.map((g) => (
            <article key={g.title} className="rounded border border-rule bg-surface p-4 sm:p-6">
              <header className="flex flex-wrap items-baseline justify-between gap-3 mb-4">
                <h3 className="text-xl text-ink">{g.title}</h3>
                <VerdictBadge verdict={verdictFor(g)} />
              </header>
              <div className={`grid gap-6 ${g.keys.length > 1 ? "lg:grid-cols-2" : ""}`}>
                {g.keys.map((k) => {
                  const c = curves[k];
                  const last = c.points[c.points.length - 1];
                  const period = k.endsWith("development") ? "Development" : k.endsWith("holdout") ? "Holdout" : null;
                  return (
                    <div key={k}>
                      {period && <p className="text-sm font-medium text-ink mb-1">{period}</p>}
                      <p className="text-sm text-muted mb-2 tabular">
                        Strategy {pct(c.cagr)}/yr · {c.points[0].date.slice(0, 7)} → {last.date.slice(0, 7)} · ends at{" "}
                        {num(last.strategy)} vs {num(last.benchmark)} ({c.benchmark})
                      </p>
                      <EquityChart
                        points={c.points}
                        benchmark={c.benchmark}
                        label={`${c.title}: growth of 1 to ${num(last.strategy)} for the strategy and ${num(last.benchmark)} for ${c.benchmark}.`}
                      />
                      <p className="text-xs text-muted mt-2">{c.note}</p>
                    </div>
                  );
                })}
              </div>
            </article>
          ))}
        </div>
      </Section>

      <Section id="ic" title="Ranking skill by year (STEP 14a, small caps)">
        <p className="text-muted mb-4 max-w-3xl">
          Rank IC is the per-date correlation between a model&rsquo;s ranking and what happened next. Zero means no
          skill. The model&rsquo;s IC was positive in {positiveYears} of {ic.length} out-of-sample years, and STEP
          14a&rsquo;s pre-registered verdict was{" "}
          <VerdictBadge verdict={verdicts.find((v) => v.id === "p1-14a-12")?.verdict ?? ""} />.
        </p>
        <div className="rounded border border-rule bg-surface p-4 sm:p-6">
          <ICByYearChart data={ic} label={`Mean rank IC by year, ${ic[0].year} to ${ic[ic.length - 1].year}, for the model and for momentum. The model is positive in ${positiveYears} of ${ic.length} years.`} />
        </div>
      </Section>

      <Section id="random" title="Percentile vs random pickers">
        <p className="text-muted mb-4 max-w-3xl">
          Each strategy against random pickers on the same dates, universe and exits. Passing needed at least the 95th
          percentile (marked). Beating most random pickers is common in a rising market; beating the market is not.
        </p>
        <ul className="space-y-2 rounded border border-rule bg-surface p-4 sm:p-6">
          {randomBaselines.map((r) => (
            <li key={`${r.step}-${r.name}`} className="grid gap-1 sm:grid-cols-[minmax(0,18rem)_1fr_3rem] sm:items-center sm:gap-4">
              <span className="text-sm text-ink">
                <span className="text-muted tabular">{r.step} · </span>
                {r.name}
              </span>
              <span className="relative h-3 rounded-sm bg-accent-soft" aria-hidden="true">
                <span className="absolute inset-y-0 left-0 rounded-sm bg-[var(--series-strategy)]" style={{ width: `${r.percentile}%` }} />
                <span className="absolute -inset-y-1 w-0.5 bg-ink" style={{ left: `${r.bar}%` }} />
              </span>
              <span className="text-sm tabular text-right">{r.percentile.toFixed(0)}</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section id="phase2" title="Phase 2 · development and one-time holdout">
        <p className="text-muted mb-4 max-w-3xl">
          A rule passes only if all five checks hold in both periods. The holdout was opened once, on{" "}
          {phase2.holdout_runs[0].time.slice(0, 10)} (commit <code className="font-mono">{phase2.holdout_runs[0].commit}</code>).
        </p>
        <div className="overflow-x-auto rounded border border-rule bg-surface">
          <table className="w-full text-sm tabular">
            <caption className="sr-only">Phase 2 results per hypothesis and period</caption>
            <thead>
              <tr className="border-b border-rule text-left text-muted">
                {["Rule", "Period", "Trades", "Mean excess / trade", "NW t", "95% CI", "Random pct", "CAGR", "Bench, same exposure", "Checks"].map((h) => (
                  <th key={h} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {phase2.rows.map((r) => (
                <tr key={`${r.hypothesis}-${r.period}`} className="border-b border-rule last:border-0">
                  <td className="px-3 py-2 whitespace-nowrap">{r.hypothesis}</td>
                  <td className="px-3 py-2">{r.period}</td>
                  <td className="px-3 py-2">{r.trades.toLocaleString("en-US")}</td>
                  <td className="px-3 py-2">{pct(r.mean_excess, 2)}</td>
                  <td className="px-3 py-2">{num(r.nw_t)}</td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    [{pct(r.ci_low, 2)}, {pct(r.ci_high, 2)}]
                  </td>
                  <td className="px-3 py-2">{r.random_pct === null ? "n/a" : r.random_pct.toFixed(0)}</td>
                  <td className="px-3 py-2">{pct(r.portfolio_cagr, 1)}</td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    {pct(r.benchmark_same_exposure_cagr, 1)} ({r.benchmark})
                  </td>
                  <td className="px-3 py-2">{r.checks_passed}/5</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="phase1b" title="Phase 1b · 11 registered variants on the holdout">
        <p className="text-muted mb-4 max-w-3xl">
          One look at 2024-01 → 2026-07. A variant needed a rank-IC t above 2 after a Holm correction over the 11, a
          positive mean IC, and a positive top-minus-bottom spread after costs. Lazy Prices (the text features):{" "}
          <VerdictBadge verdict={phase1b.lazy_prices} />
        </p>
        <div className="overflow-x-auto rounded border border-rule bg-surface">
          <table className="w-full text-sm tabular">
            <caption className="sr-only">Phase 1b holdout results per variant</caption>
            <thead>
              <tr className="border-b border-rule text-left text-muted">
                {["Model", "Features", "Mean rank IC", "NW t", "Holm p", "Spread / month", "Top decile / yr", "SPY / yr", "Verdict"].map((h) => (
                  <th key={h} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {phase1b.rows.map((r) => (
                <tr key={`${r.model}-${r.features}`} className="border-b border-rule last:border-0">
                  <td className="px-3 py-2 whitespace-nowrap">{r.model}</td>
                  <td className="px-3 py-2 whitespace-nowrap">{r.features}</td>
                  <td className="px-3 py-2">{num(r.mean_rank_ic, 4)}</td>
                  <td className="px-3 py-2">{num(r.nw_t)}</td>
                  <td className="px-3 py-2">{num(r.holm_p, 3)}</td>
                  <td className="px-3 py-2">{pct(r.spread_mean_monthly, 2)}</td>
                  <td className="px-3 py-2">{pct(r.top_decile_annualized, 1)}</td>
                  <td className="px-3 py-2">{pct(r.spy_annualized, 1)}</td>
                  <td className="px-3 py-2">
                    <VerdictBadge verdict={r.verdict} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}
