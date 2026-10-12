import type { Metadata } from "next";
import { PageHeader, Section, VerdictBadge } from "@/components/ui";
import { ledger, num, pct, phase3Row, shortDate } from "@/lib/data";

export const metadata: Metadata = { title: "Live ledger" };

function NotStarted() {
  return (
    <div className="rounded border border-dashed border-rule bg-surface p-6 sm:p-8 max-w-3xl">
      <p className="text-lg text-ink mb-3">
        {ledger.forward_test_start
          ? `Forward test starts on ${ledger.forward_test_start}.`
          : "The forward test has not started yet."}
      </p>
      <p className="text-muted mb-3">Status: {phase3Row?.test}.</p>
      <p className="text-muted">
        When it starts, every week records picks from four arms on the same candidate list: random picks (seeded, so
        anyone can reproduce them), the ML ranker (only if it passed its backtest), an LLM research agent (its buys and
        its passes), and manual picks. Entries are filled at the next day&rsquo;s open, and each record stores the hash
        of the previous one, so any later edit breaks the chain. An arm is judged only after at least 26 weeks and 60
        closed picks, against IWM and against the random arm.
      </p>
    </div>
  );
}

export default function LedgerPage() {
  const picks = [...ledger.picks].sort((a, b) => b.week.localeCompare(a.week));
  return (
    <div>
      <PageHeader eyebrow="Phase 3" title="Live ledger">
        The only test left that no backtest has seen: the future, recorded week by week before the outcome is known.
      </PageHeader>

      {ledger.status === "not_started" ? (
        <NotStarted />
      ) : (
        <>
          <Section title="Ledger integrity">
            <p className="text-muted">
              Hash chain:{" "}
              <VerdictBadge verdict={ledger.chain_ok ? "CHAIN VERIFIED" : "CHAIN BROKEN"} /> · updated{" "}
              {shortDate(ledger.generated_at)}
              {ledger.forward_test_start && ` · forward test started ${ledger.forward_test_start}`}
            </p>
          </Section>
          <Section title="Arms">
            <div className="overflow-x-auto rounded border border-rule bg-surface">
              <table className="w-full text-sm tabular">
                <caption className="sr-only">Forward results per arm</caption>
                <thead>
                  <tr className="border-b border-rule text-left text-muted">
                    {["Arm", "Picks", "Closed", "Mean excess vs IWM", "95% CI", "Pct vs random arm", "Verdict"].map((h) => (
                      <th key={h} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ledger.arms.map((a) => (
                    <tr key={a.name} className="border-b border-rule last:border-0">
                      <td className="px-3 py-2">{a.name}</td>
                      <td className="px-3 py-2">{a.picks}</td>
                      <td className="px-3 py-2">{a.closed}</td>
                      <td className="px-3 py-2">{pct(a.mean_excess_iwm, 2)}</td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        [{pct(a.ci_low, 2)}, {pct(a.ci_high, 2)}]
                      </td>
                      <td className="px-3 py-2">{num(a.random_pct, 0)}</td>
                      <td className="px-3 py-2">
                        <VerdictBadge verdict={a.verdict ?? "NOT PROVEN"} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
          <Section title="Recorded picks">
            <p className="text-muted mb-4 max-w-3xl">
              A record of past decisions for the forward test, not recommendations. Most will not beat the index.
            </p>
            <div className="overflow-x-auto rounded border border-rule bg-surface">
              <table className="w-full text-sm tabular">
                <caption className="sr-only">Every recorded pick, newest week first</caption>
                <thead>
                  <tr className="border-b border-rule text-left text-muted">
                    {["Week", "Arm", "Ticker", "Entry", "Status", "Exit", "Excess vs IWM"].map((h) => (
                      <th key={h} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {picks.map((p, i) => (
                    <tr key={`${p.week}-${p.arm}-${p.ticker}-${i}`} className="border-b border-rule last:border-0">
                      <td className="px-3 py-2 whitespace-nowrap">{p.week}</td>
                      <td className="px-3 py-2">{p.arm}</td>
                      <td className="px-3 py-2 font-mono">{p.ticker}</td>
                      <td className="px-3 py-2 whitespace-nowrap">{p.entry_date ?? "pending"}</td>
                      <td className="px-3 py-2">{p.status}</td>
                      <td className="px-3 py-2 whitespace-nowrap">{p.exit_date ?? "--"}</td>
                      <td className="px-3 py-2">{pct(p.excess_iwm, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </>
      )}
    </div>
  );
}
