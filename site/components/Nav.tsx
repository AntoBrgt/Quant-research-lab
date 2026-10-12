"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/methodology/", label: "Methodology" },
  { href: "/results/", label: "Results" },
  { href: "/preregistrations/", label: "Pre-registrations" },
  { href: "/ledger/", label: "Live ledger" },
  { href: "/about/", label: "About" },
];

function isCurrent(pathname: string, href: string): boolean {
  const p = pathname.endsWith("/") ? pathname : `${pathname}/`;
  return href === "/" ? p === "/" : p.startsWith(href);
}

export function Nav() {
  const pathname = usePathname() || "/";
  const [open, setOpen] = useState(false);
  return (
    <header className="border-b border-rule bg-paper">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        <Link href="/" className="font-serif text-lg text-ink no-underline whitespace-nowrap">
          Quant Research Lab
        </Link>
        <nav aria-label="Main" className="hidden md:block">
          <ul className="flex items-center gap-5 text-sm">
            {LINKS.map((l) => (
              <li key={l.href}>
                <Link
                  href={l.href}
                  aria-current={isCurrent(pathname, l.href) ? "page" : undefined}
                  className={
                    isCurrent(pathname, l.href)
                      ? "text-ink font-medium underline decoration-2 underline-offset-[6px]"
                      : "text-muted no-underline hover:text-ink"
                  }
                >
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <button
            type="button"
            className="md:hidden h-9 px-3 rounded border border-rule text-sm text-ink"
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((v) => !v)}
          >
            Menu
          </button>
        </div>
      </div>
      <nav id="mobile-nav" aria-label="Main" className={`md:hidden border-t border-rule ${open ? "block" : "hidden"}`}>
        <ul className="max-w-5xl mx-auto px-4 py-2">
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link
                href={l.href}
                onClick={() => setOpen(false)}
                aria-current={isCurrent(pathname, l.href) ? "page" : undefined}
                className={`block py-2.5 no-underline ${isCurrent(pathname, l.href) ? "text-ink font-medium" : "text-muted"}`}
              >
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
