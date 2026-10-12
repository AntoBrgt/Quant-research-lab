// Post-build fix for the static export. Next 16 writes nested route segments as
// out/<route>/__next.<route>/__PAGE__.txt but the client router requests the flat name
// out/<route>/__next.<route>.__PAGE__.txt. A static host (Vercel static, GitHub Pages)
// can't map one to the other, so prefetches 404 and pages log console errors.
// This copies every nested segment file to the flat name the router asks for.
import { copyFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";

const OUT = new URL("../out/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");

function walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path);
    else flatten(path);
  }
}

function flatten(file) {
  const parts = relative(OUT, file).split(sep);
  const i = parts.findIndex((p, k) => p.startsWith("__next.") && k < parts.length - 1);
  if (i < 0) return;
  const flat = join(OUT, ...parts.slice(0, i), parts.slice(i).join("."));
  copyFileSync(file, flat);
  copied += 1;
}

let copied = 0;
walk(OUT);
console.log(`flatten-segments: ${copied} segment file(s) copied to flat names`);
