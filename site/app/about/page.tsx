import type { Metadata } from "next";
import { PageHeader, Section } from "@/components/ui";
import { lessons } from "@/lib/data";

export const metadata: Metadata = { title: "About" };

const STACK = [
  ["Research", "Python 3.12, pandas, NumPy, SciPy, LightGBM, XGBoost, lifelines, scikit-learn, sentence-transformers"],
  ["Data", "SEC EDGAR (13F, Form 4, XBRL, 10-K/10-Q text), OpenFIGI, Yahoo Finance prices, all cached point in time"],
  ["Testing", "pytest, 380+ tests for look-ahead, embargo, costs, holdout locks and verdict rules; CI on every push"],
  ["This site", "Next.js static export, TypeScript, Tailwind CSS, Recharts; no backend, data from versioned JSON"],
];

export default function About() {
  return (
    <div>
      <PageHeader eyebrow="About" title="Antonin Brengetto" />

      <section className="mb-14 max-w-3xl space-y-4 text-lg text-ink">
        <p>
          I&rsquo;m a data scientist. I invest some of my own money every month, and I wanted a tool to help me decide
          what to buy. So I built one, then checked whether its picks would really have beaten the index: only data
          available at the time, trading costs included, and the rules of each test written down before running it.
          They didn&rsquo;t, and neither did any model or rule I tried after that.
        </p>
        <p>
          What surprised me most is how hard beating the market is, and that reproducing what institutions do is out of
          reach for one person working alone. If I started again, I would tackle a different question: is my portfolio
          safe?
        </p>
        <p className="text-base text-muted">
          Built with Claude Code as a coding assistant; research design, pre-registrations and verdicts reviewed and
          decided by me.
        </p>
      </section>

      <Section title="What I learned">
        <ul className="max-w-3xl list-disc pl-5 space-y-3 text-muted">
          {lessons.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </Section>

      <Section title="Stack">
        <dl className="max-w-3xl divide-y divide-[var(--rule)] rounded border border-rule bg-surface">
          {STACK.map(([k, v]) => (
            <div key={k} className="grid gap-1 sm:grid-cols-[9rem_1fr] p-4">
              <dt className="font-medium text-ink">{k}</dt>
              <dd className="text-muted">{v}</dd>
            </div>
          ))}
        </dl>
      </Section>
    </div>
  );
}
