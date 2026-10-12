import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { meta, REPO_URL } from "@/lib/data";

export const metadata: Metadata = {
  title: { default: "Quant Research Lab", template: "%s · Quant Research Lab" },
  description:
    "Can machine learning pick stocks that beat the index once the test is honest? Point-in-time data, costs, walk-forward, random baselines and pre-registered verdicts. Answer so far: no.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fbfaf7" },
    { media: "(prefers-color-scheme: dark)", color: "#121417" },
  ],
};

// Runs before first paint: stored choice, else the system preference. No flash, no layout shift.
const themeScript = `(function(){try{var t=localStorage.getItem("theme");if(t!=="light"&&t!=="dark"){t=window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"}document.documentElement.setAttribute("data-theme",t)}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className="min-h-screen flex flex-col">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 bg-surface px-3 py-2 rounded border border-rule"
        >
          Skip to content
        </a>
        <Nav />
        <main id="main" className="flex-1 w-full max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
          {children}
        </main>
        <footer className="border-t border-rule">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 text-sm text-muted space-y-2">
            <p className="font-medium text-ink">
              Research project. Not investment advice. No strategy shown here has beaten the market.
            </p>
            <p>
              Data exported from commit{" "}
              <a href={`${REPO_URL}/commit/${meta.code_commit}`}>
                <code className="font-mono">{meta.code_commit}</code>
              </a>{" "}
              on {meta.generated_at.slice(0, 10)} · <a href={REPO_URL}>Source on GitHub</a>
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
