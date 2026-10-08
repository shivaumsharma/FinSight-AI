import { expect, test } from "@playwright/test";
import { writeFileSync } from "node:fs";
import path from "node:path";
import type { BenchResult, Mode } from "../src/app/bench/BenchClient";

// Runs the three configurations in fresh page loads (fresh heaps) against a production build and records the numbers.
// Results: bench/results.json. The shipped configuration ("full") must do far less React work than the naive one.

const MODES: Mode[] = ["naive", "batched", "full"];

test("fast price feed: batching and memoized rows cut React render work", async ({ browser }) => {
  const results: BenchResult[] = [];
  for (const mode of MODES) {
    const page = await browser.newPage();
    await page.goto("/bench");
    await page.waitForFunction(() => typeof (window as unknown as { __runBench?: unknown }).__runBench === "function");
    const result = await page.evaluate((m) => (window as unknown as { __runBench: (m: string) => Promise<BenchResult> }).__runBench(m), mode);
    results.push(result);
    await page.close();
  }

  const by = Object.fromEntries(results.map((r) => [r.mode, r])) as Record<Mode, BenchResult>;
  const summary = {
    measuredAt: new Date().toISOString(),
    environment: "production build with React profiling, headless Chromium, one machine; relative numbers are the point",
    workload: `${by.naive.rows} rows, ${by.naive.tickRatePerSecond} ticks/s for ${by.naive.durationMs / 1000}s, uniformly random tickers`,
    results,
    reduction: {
      reactRenderMs: Number((by.naive.reactRenderMs / by.full.reactRenderMs).toFixed(1)),
      rowRenders: Number((by.naive.rowRenders / by.full.rowRenders).toFixed(1)),
      commits: Number((by.naive.commits / by.full.commits).toFixed(1)),
    },
  };
  writeFileSync(path.join(__dirname, "results.json"), JSON.stringify(summary, null, 2) + "\n");

  console.table(results.map((r) => ({ mode: r.mode, commits: r.commits, rowRenders: r.rowRenders, reactRenderMs: r.reactRenderMs, p95CommitMs: r.p95CommitMs, maxCommitMs: r.maxCommitMs, longTasks: r.longTasks })));
  console.log("reduction naive -> full:", summary.reduction);

  expect(by.full.rowRenders).toBeLessThan(by.naive.rowRenders / 5);
  expect(by.full.reactRenderMs).toBeLessThan(by.naive.reactRenderMs / 3);
  expect(by.full.commits).toBeLessThan(by.naive.commits / 5);
});
