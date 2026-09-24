#!/usr/bin/env node
// On-demand external-link checker for book/**/*.md.
// Fetches every http(s) URL and classifies it:
//   OK        2xx / 3xx
//   BLOCKED   401/403/405/406/429 (bot-block or auth wall; the page almost certainly exists)
//   DEAD      404/410, DNS failure, TLS error, or connection refused  <- the ones to fix
//
// This is deliberately NOT wired into CI: external sites rate-limit and bot-block,
// which would make a push gate flaky. Run it manually or on a schedule:
//   node tools/check-links.mjs
// Exits non-zero if any DEAD link is found.

import { readFileSync, readdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const ROOT = "book";
// papers.md and datasets.md are checked too: they are a couple of hundred external
// links and nothing else, so link rot there is the whole failure mode. The case-study
// indexes stay out on purpose; those point at hundreds of company blogs that get
// reorganized constantly, and one 404 should not turn this job red every month.
const FILES = ["papers.md", "datasets.md"].filter((f) => existsSync(f));
const CONCURRENCY = 12;
const TIMEOUT_MS = 20000;
const UA = "Mozilla/5.0 (compatible; neurarch-linkcheck/1.0)";

function walk(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith(".md")) out.push(p);
  }
  return out;
}

// url -> Set of files that reference it
const refs = new Map();
for (const f of [...walk(ROOT), ...FILES]) {
  const t = readFileSync(f, "utf8");
  for (const m of t.matchAll(/\]\((https?:\/\/[^)\s]+)\)/g)) {
    const u = m[1];
    if (!refs.has(u)) refs.set(u, new Set());
    refs.get(u).add(f);
  }
}
const urls = [...refs.keys()];
console.log(`Checking ${urls.length} unique external URLs from ${[`${ROOT}/`, ...FILES].join(", ")} ...\n`);

const BLOCKED = new Set([401, 403, 405, 406, 429]);

// Hosts that answer a scripted client with 404 rather than 403 when they feel like it.
// Kaggle does this intermittently: the same competition URL returns 200 to a browser
// and 404 here, and it is not consistent between runs, so treating those as DEAD would
// make this job red most months for links that are fine. The cost is real and worth
// naming: for these hosts the checker cannot tell rot from a block, so a link to one of
// them has to be opened by hand when it is added.
const FLAKY_404_HOSTS = ["www.kaggle.com", "kaggle.com"];
const isFlakyHost = (u) => {
  try {
    return FLAKY_404_HOSTS.includes(new URL(u).host);
  } catch {
    return false;
  }
};

async function probe(u) {
  for (const method of ["HEAD", "GET"]) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const r = await fetch(u, { method, redirect: "follow", signal: ctrl.signal, headers: { "User-Agent": UA } });
      clearTimeout(timer);
      if (method === "HEAD" && (r.status === 405 || r.status === 501)) continue; // retry with GET
      if (r.status >= 200 && r.status < 400) return { cls: "OK", status: r.status };
      if (BLOCKED.has(r.status)) return { cls: "BLOCKED", status: r.status };
      if (r.status === 404 && isFlakyHost(u)) return { cls: "BLOCKED", status: "404 (host blocks scripts)" };
      return { cls: "DEAD", status: r.status };
    } catch (e) {
      clearTimeout(timer);
      if (method === "HEAD") continue; // some hosts refuse HEAD; try GET
      return { cls: "DEAD", status: e.cause?.code || e.name || "fetch-error" };
    }
  }
  return { cls: "DEAD", status: "no-response" };
}

const results = [];
let i = 0;
async function worker() {
  while (i < urls.length) {
    const u = urls[i++];
    results.push({ u, ...(await probe(u)) });
  }
}
await Promise.all(Array.from({ length: CONCURRENCY }, worker));

const dead = results.filter((r) => r.cls === "DEAD");
const blocked = results.filter((r) => r.cls === "BLOCKED");
console.log(`OK: ${results.length - dead.length - blocked.length}   BLOCKED (likely live): ${blocked.length}   DEAD: ${dead.length}\n`);
if (dead.length) {
  console.log("DEAD links (fix these):");
  for (const d of dead.sort((a, b) => String(a.status).localeCompare(String(b.status)))) {
    const where = [...refs.get(d.u)].join(", ");
    console.log(`  [${d.status}] ${d.u}\n        in: ${where}`);
  }
  process.exit(1);
}
console.log("No dead links.");
