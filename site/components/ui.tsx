import Link from "next/link";

/** Sober verdict badge: tone by meaning, never trading green/red. */
export function VerdictBadge({ verdict }: { verdict: string }) {
  const v = verdict.toUpperCase();
  const pending = v.includes("NOT STARTED") || v.includes("DEVELOPMENT ONLY");
  const passed = v === "PASS" || v === "PASSED" || v === "TRADEABLE" || v === "PROMOTED" || v === "PROMISING";
  const cls = pending
    ? "border-dashed border-rule text-muted"
    : passed
      ? "border-accent text-accent bg-accent-soft"
      : "border-rule text-ink bg-surface";
  return (
    <span className={`inline-block whitespace-nowrap rounded border px-2 py-0.5 text-xs font-medium tracking-wide ${cls}`}>
      {verdict}
    </span>
  );
}

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: React.ReactNode }) {
  return (
    <header className="mb-10 max-w-3xl">
      {eyebrow && <p className="text-sm uppercase tracking-widest text-muted mb-2">{eyebrow}</p>}
      <h1 className="text-3xl sm:text-4xl text-ink">{title}</h1>
      {children && <div className="mt-4 text-lg text-muted">{children}</div>}
    </header>
  );
}

export function Section({ id, title, children }: { id?: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mb-14 scroll-mt-20">
      <h2 className="text-2xl text-ink mb-4">{title}</h2>
      {children}
    </section>
  );
}

export function ButtonLink({ href, children, primary }: { href: string; children: React.ReactNode; primary?: boolean }) {
  return (
    <Link
      href={href}
      className={`inline-flex items-center h-11 px-5 rounded no-underline text-sm font-medium border ${
        primary ? "bg-accent text-paper border-accent" : "border-rule text-ink bg-surface"
      }`}
    >
      {children}
    </Link>
  );
}
