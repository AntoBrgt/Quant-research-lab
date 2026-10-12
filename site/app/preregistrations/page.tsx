import type { Metadata } from "next";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PageHeader } from "@/components/ui";
import { preregistrations, REPO_URL, shortDate, type Commit } from "@/lib/data";

export const metadata: Metadata = { title: "Pre-registrations" };

function CommitLink({ commit, label }: { commit: Commit | null | undefined; label: string }) {
  if (!commit) return null;
  return (
    <span className="whitespace-nowrap">
      {label}{" "}
      <a href={commit.url}>
        <code className="font-mono">{commit.hash}</code>
      </a>{" "}
      <span className="text-muted">({shortDate(commit.date)})</span>
    </span>
  );
}

export default function Preregistrations() {
  return (
    <div>
      <PageHeader eyebrow="Pre-registrations" title="The rules, frozen before the tests">
        Each document below was committed to git before the test it governs touched any data. The commit link is the
        proof: its date comes before the results, and the history shows whether the text changed afterwards.
      </PageHeader>

      <nav aria-label="Documents" className="mb-10">
        <ul className="flex flex-wrap gap-3 text-sm">
          {preregistrations.map((d) => (
            <li key={d.id}>
              <a href={`#${d.id}`} className="inline-block rounded border border-rule bg-surface px-3 py-1.5 no-underline text-ink">
                {d.title}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      <div className="space-y-16">
        {preregistrations.map((d) => (
          <article key={d.id} id={d.id} className="scroll-mt-20">
            <div className="mb-6 rounded border border-rule bg-surface p-4 text-sm space-y-1">
              <p className="font-medium text-ink">{d.source}</p>
              <p className="flex flex-wrap gap-x-6 gap-y-1">
                <CommitLink commit={d.frozen_commit} label="Frozen in commit" />
                <CommitLink commit={d.protocol_commit} label="Protocol first committed in" />
              </p>
              {d.unchanged_since_freeze !== null && (
                <p className="text-muted">
                  {d.unchanged_since_freeze
                    ? "Unchanged since that commit."
                    : "Changed after the freeze commit; see the file history on GitHub."}{" "}
                  <a href={`${REPO_URL}/commits/main/${d.source}`}>File history</a>
                </p>
              )}
            </div>
            <div className="prose-doc max-w-3xl">
              <Markdown remarkPlugins={[remarkGfm]}>{d.markdown}</Markdown>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
