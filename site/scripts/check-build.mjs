// Post-build check (CI): the built HTML shows exactly what the exported JSON says.
// - every page carries the disclaimer footer;
// - every verdict row (name, result, benchmark, verdict) appears on /results/;
// - no page except the quoted pre-registrations contains BUY/SELL wording.
import { readFileSync, existsSync } from "node:fs";

const OUT = new URL("../out/", import.meta.url);
const read = (p) => readFileSync(new URL(p, OUT), "utf8");
const verdicts = JSON.parse(readFileSync(new URL("../public/data/verdicts.json", import.meta.url), "utf8")).rows;
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const DISCLAIMER = "Research project. Not investment advice. No strategy shown here has beaten the market.";
const PAGES = ["index.html", "methodology/index.html", "results/index.html", "preregistrations/index.html", "ledger/index.html", "about/index.html"];

const errors = [];
for (const page of PAGES) {
  if (!existsSync(new URL(page, OUT))) { errors.push(`missing page ${page}`); continue; }
  const html = read(page);
  if (!html.includes(DISCLAIMER)) errors.push(`${page}: disclaimer footer missing`);
  if (page !== "preregistrations/index.html") {
    const text = html.replace(/<script[\s\S]*?<\/script>/g, "").replace(/<[^>]+>/g, " ");
    if (/\b(BUY|SELL)\b/.test(text)) errors.push(`${page}: contains BUY/SELL wording`);
  }
}
const results = read("results/index.html");
for (const r of verdicts) {
  for (const field of ["name", "annualized_after_costs", "benchmark_same_period", "verdict"]) {
    if (!results.includes(esc(r[field]))) errors.push(`results: ${r.id} ${field} not shown: ${r[field]}`);
  }
}
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log(`check-build: ${PAGES.length} pages, ${verdicts.length} verdict rows -- all present`);
