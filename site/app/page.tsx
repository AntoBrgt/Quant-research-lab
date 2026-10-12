import { ButtonLink, VerdictBadge } from "@/components/ui";
import { phase1b, phase2, phase3Row, testedRows } from "@/lib/data";

export default function Home() {
  const beat = testedRows.filter((r) => /^(PASS|PASSED|TRADEABLE|PROMOTED)$/i.test(r.verdict)).length;
  const holdouts = phase2.holdout_runs.length + (phase1b.holdout_looks ? 1 : 0);
  const keyNumbers = [
    { value: String(testedRows.length), label: "strategies and models tested to a recorded verdict" },
    { value: String(beat), label: "beat the index after costs on data they had never seen" },
    { value: String(holdouts), label: `one-time holdouts opened exactly once (each in a logged, locked run)` },
  ];
  return (
    <div>
      <section className="max-w-3xl mb-14">
        <p className="text-sm uppercase tracking-widest text-muted mb-3">Quantitative research · 2013–2026 US equities</p>
        <h1 className="text-3xl sm:text-5xl text-ink mb-6">
          Can machine learning pick stocks that beat the index, once the test is honest?
        </h1>
        <p className="text-lg sm:text-xl text-ink mb-4">
          <strong>Not so far.</strong> Across three research phases, I tested ranking models, trading loops, value and
          survival models, a small-cap model, filing-text features and three event rules. None of them beat SPY or IWM
          after costs.
        </p>
        <p className="text-muted mb-8">
          Several beat most random stock pickers, and a few had good stretches. None cleared its pre-registered bar on
          data it had never seen. Every number on this site comes from
          the published research log.
        </p>
        <div className="flex flex-wrap gap-3">
          <ButtonLink href="/methodology/" primary>
            How a test is run
          </ButtonLink>
          <ButtonLink href="/results/">See every result</ButtonLink>
        </div>
      </section>

      <section aria-labelledby="key-numbers" className="mb-14">
        <h2 id="key-numbers" className="sr-only">
          Key numbers
        </h2>
        <dl className="grid gap-4 sm:grid-cols-3">
          {keyNumbers.map((k) => (
            <div key={k.label} className="rounded border border-rule bg-surface p-5">
              <dt className="sr-only">{k.label}</dt>
              <dd className="font-serif text-5xl text-ink tabular">{k.value}</dd>
              <dd className="mt-2 text-sm text-muted">{k.label}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="max-w-3xl">
        <h2 className="text-2xl text-ink mb-4">What is next</h2>
        <p className="text-muted mb-3">
          Every month since 2013 has now been used by some test, so another backtest can only filter ideas. The last
          clean test is the future: Phase 3 records picks every week in an append-only, hash-chained ledger and judges
          them after at least 26 weeks, against random picks from the same candidates.
        </p>
        <p className="text-sm">
          Phase 3 status: <VerdictBadge verdict={phase3Row?.verdict ?? "NOT STARTED"} />{" "}
          <span className="text-muted">{phase3Row?.test}</span>
        </p>
      </section>
    </div>
  );
}
