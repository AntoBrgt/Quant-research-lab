import type { Metadata } from "next";
import { PageHeader, Section } from "@/components/ui";
import { LOG_URL, REPO_URL } from "@/lib/data";

export const metadata: Metadata = { title: "About" };

// TODO(Antonin): replace the placeholder bio and the LinkedIn URL.
const LINKEDIN_URL = "";

const STACK = [
  ["Research", "Python 3.12, pandas, NumPy, SciPy, LightGBM, XGBoost, lifelines, scikit-learn, sentence-transformers"],
  ["Data", "SEC EDGAR (13F, Form 4, XBRL, 10-K/10-Q text), OpenFIGI, Yahoo Finance prices, all cached point in time"],
  ["Testing", "pytest, 380+ tests for look-ahead, embargo, costs, holdout locks and verdict rules; CI on every push"],
  ["This site", "Next.js static export, TypeScript, Tailwind CSS, Recharts; no backend, data from versioned JSON"],
];

export default function About() {
  return (
    <div>
      <PageHeader eyebrow="About" title="Antonin Brengetto">
        I build and test quantitative research pipelines: getting the data right, and not fooling myself about the
        results.
      </PageHeader>

      <Section title="About this project">
        <div className="max-w-3xl space-y-3 text-muted">
          <p>
            Quant Research Lab started as a question: could a machine-learning model, or any systematic rule on public
            data, beat the index after costs? It became a lesson in how backtests mislead, and in building the safeguards
            that stop them from doing so.
          </p>
          <p>
            The answer so far is no. That negative result is the product: a reproducible, pre-registered record of what
            does not work, and why each test can be trusted.
          </p>
        </div>
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

      <Section title="Links">
        <ul className="space-y-2">
          <li>
            <a href={REPO_URL}>Source code on GitHub</a>
          </li>
          <li>
            <a href={LOG_URL}>Full research log</a>
          </li>
          <li>
            {LINKEDIN_URL ? <a href={LINKEDIN_URL}>LinkedIn</a> : <span className="text-muted">LinkedIn (link to be added)</span>}
          </li>
        </ul>
      </Section>
    </div>
  );
}
