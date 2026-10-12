import type { Metadata } from "next";
import { PageHeader, Section } from "@/components/ui";
import { LOG_URL } from "@/lib/data";

export const metadata: Metadata = { title: "Methodology" };

const PIPELINE = [
  {
    title: "Point-in-time universe",
    text: "Each month, only stocks that were knowable then, rebuilt from SEC 13F filings by filing date. Companies that later failed or were bought stay in, when they can be priced.",
  },
  {
    title: "Features, dated by publication",
    text: "Fundamentals from SEC XBRL and insider trades from Form 4, each usable only from the day it was filed.",
  },
  {
    title: "Walk-forward with an embargo",
    text: "Models train on the past, are tested on the next block, then retrain. A gap at least as long as the label keeps training labels from overlapping test dates.",
  },
  {
    title: "Trading costs",
    text: "Entry at the next day's open, never the signal-day close. 0.15% per side for large caps, 0.30% for small caps.",
  },
  {
    title: "Random baseline",
    text: "Hundreds of random pickers on the same dates, universe and exits. Beating them is necessary, not sufficient.",
  },
  {
    title: "Pre-registered verdict",
    text: "The pass rule is committed to git before the test runs, and nothing is tuned on test data.",
  },
  {
    title: "One-time holdout",
    text: "Untouched recent data, opened once by a locked, logged run. A second look would turn it into more development data.",
  },
];

export default function Methodology() {
  return (
    <div>
      <PageHeader eyebrow="Methodology" title="How a test is run">
        Most backtests that look good are wrong for a few well-known reasons. Each step below removes one of them.
      </PageHeader>

      <Section title="The pipeline">
        <ol className="grid gap-3 md:grid-cols-7 md:gap-2" aria-label="Test pipeline, in order">
          {PIPELINE.map((step, i) => (
            <li key={step.title} className="relative rounded border border-rule bg-surface p-4 md:p-3">
              <p className="text-xs text-muted tabular mb-1">Step {i + 1}</p>
              <h3 className="text-base md:text-sm font-semibold text-ink mb-1">{step.title}</h3>
              <p className="text-sm md:text-xs text-muted leading-relaxed">{step.text}</p>
              {i < PIPELINE.length - 1 && (
                <span aria-hidden="true" className="absolute text-muted md:-right-2 md:top-1/2 md:-translate-y-1/2 left-1/2 -bottom-3 md:left-auto md:bottom-auto">
                  <span className="md:hidden">↓</span>
                  <span className="hidden md:inline">→</span>
                </span>
              )}
            </li>
          ))}
        </ol>
      </Section>

      <div className="grid gap-10 md:grid-cols-2">
        <Section title="Look-ahead bias, in plain words">
          <p className="text-muted mb-3">
            Look-ahead bias means using information in a test that nobody had at the time. It is easy to do by accident:
            a quarterly report describes March but is published in May. A test that uses it on March 31 is reading the
            future.
          </p>
          <p className="text-muted">
            Here every input is dated by when it became public: SEC filing dates, the earliest filed version of each
            number (not later corrections), and entry at the next open. Tests check that removing later data leaves
            earlier features unchanged. No large language model is used in any backtest, because it may have read what
            happened next.
          </p>
        </Section>
        <Section title="Survivorship bias, in plain words">
          <p className="text-muted mb-3">
            Testing on today&rsquo;s index members means testing only on the companies that survived. The failures,
            acquisitions and delistings drop out, and almost any strategy looks good.
          </p>
          <p className="text-muted">
            Here the universe is rebuilt month by month from what institutions actually held at the time. When old
            securities cannot be priced, the gap is measured and reported, with worst-case bounds, rather than ignored.
          </p>
        </Section>
      </div>

      <Section title="How results are judged">
        <ul className="list-disc pl-5 space-y-2 text-muted max-w-3xl">
          <li>
            <strong className="text-ink">Newey-West t-statistics</strong> on per-date rank correlations and on monthly
            excess returns. Overlapping holding periods would otherwise overstate the evidence.
          </li>
          <li>
            <strong className="text-ink">Bootstrap confidence intervals</strong>, resampled by month, because trades in
            the same month are not independent.
          </li>
          <li>
            <strong className="text-ink">Multiple-testing corrections</strong> (Holm) when several variants are judged
            at once, and a registry that counts every variant ever tried.
          </li>
          <li>
            <strong className="text-ink">The benchmark at the same exposure</strong>: a strategy that is 40% invested is
            compared with 40% in the index, not 100%.
          </li>
          <li>
            <strong className="text-ink">Bugs found before a holdout are fixed and reported</strong>, with their effect
            on the result, before the holdout is opened.
          </li>
        </ul>
        <p className="mt-6 text-sm">
          Full detail, step by step: <a href={LOG_URL}>the research log</a>.
        </p>
      </Section>
    </div>
  );
}
