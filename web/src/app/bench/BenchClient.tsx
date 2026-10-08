"use client";

import { Profiler, useCallback, useEffect, useRef, useState, type ProfilerOnRenderCallback } from "react";
import { flushSync } from "react-dom";
import WatchlistRow, { WatchlistRowUnmemoized, renderProbe } from "@/components/WatchlistRow";
import { LivePriceStore, type LivePrices } from "@/lib/livePrices";
import type { PriceStreamOptions, Tick } from "@/lib/priceStream";
import { useLivePrices } from "@/lib/livePrices";
import type { WatchlistItem } from "@/lib/types";

// Same workload through three ways of handling a fast price feed, all rendering the real watchlist row:
//   naive    one render per tick, new row objects, rows not memoized        (how a first implementation behaves)
//   batched  ticks coalesced into one update per 100 ms, rows not memoized
//   full     batched + unchanged rows keep their identity + memoized rows   (what ships)
export type Mode = "naive" | "batched" | "full";

export interface BenchResult {
  mode: Mode;
  rows: number;
  ticksSent: number;
  tickRatePerSecond: number;
  durationMs: number;
  commits: number;
  rowRenders: number;
  reactRenderMs: number;
  maxCommitMs: number;
  p95CommitMs: number;
  longTasks: number;
  longTaskMs: number;
}

const ROWS = 100;
const TICK_RATE = 500; // ticks per second, spread uniformly over all rows (the hardest case for memoization)
const DURATION_MS = 4000;
const INTERVAL_MS = 10;

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const items: WatchlistItem[] = Array.from({ length: ROWS }, (_, i) => ({
  ticker: `T${String(i).padStart(3, "0")}`,
  price: 100 + i,
  change_pct: 0,
  currency: "USD",
  rating: i % 3 === 0 ? "Buy" : i % 3 === 1 ? "Hold" : "Sell",
  added_at: 0,
  next_earnings_date: null,
  next_ex_dividend_date: null,
  last_dividend_amount: null,
  last_split: null,
}));

function percentile(sorted: number[], p: number) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

// Rows driven by the real store and hook (batched / full).
function StoreList({ store, memo, onRender }: { store: LivePriceStore; memo: boolean; onRender: ProfilerOnRenderCallback }) {
  const live = useLivePrices(items.map((i) => i.ticker), store);
  const Row = memo ? WatchlistRow : WatchlistRowUnmemoized;
  return (
    <Profiler id="list" onRender={onRender}>
      <div>
        {items.map((item) => (
          <Row key={item.ticker} item={item} live={live.prices[item.ticker]} />
        ))}
      </div>
    </Profiler>
  );
}

// Rows driven by plain React state, one update per tick (naive).
function NaiveList({ prices, onRender }: { prices: LivePrices; onRender: ProfilerOnRenderCallback }) {
  return (
    <Profiler id="list" onRender={onRender}>
      <div>
        {items.map((item) => (
          <WatchlistRowUnmemoized key={item.ticker} item={item} live={prices[item.ticker] ? { ...prices[item.ticker] } : undefined} />
        ))}
      </div>
    </Profiler>
  );
}

export default function BenchClient() {
  const [mode, setMode] = useState<Mode | null>(null);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [naivePrices, setNaivePrices] = useState<LivePrices>({});
  const storeRef = useRef<LivePriceStore | null>(null);
  const feedRef = useRef<((ticks: Tick[]) => void) | null>(null);
  const statsRef = useRef({ durations: [] as number[], rowRenders: 0 });

  const onRender: ProfilerOnRenderCallback = useCallback((_id, phase, actualDuration) => {
    if (phase !== "mount") statsRef.current.durations.push(actualDuration);
  }, []);

  const run = useCallback(async (selected: Mode): Promise<BenchResult> => {
    statsRef.current = { durations: [], rowRenders: 0 };
    renderProbe.onRender = null;
    setResult(null);

    if (selected !== "naive") {
      storeRef.current = new LivePriceStore({
        flushMs: 100,
        getEndpoint: async () => ({ token: "bench", ws_url: "ws://unused" }),
        createStream: (o: PriceStreamOptions) => {
          feedRef.current = o.onTicks;
          return { start() {}, stop() {}, setTickers() {}, reconnectNow() {} };
        },
      });
    }
    setMode(selected);
    await new Promise((r) => setTimeout(r, 400)); // mount and settle
    statsRef.current.durations = [];

    const rand = mulberry32(42);
    const prices = items.map((i) => i.price ?? 0);
    let sent = 0;
    const longTasks: number[] = [];
    const observer = new PerformanceObserver((list) => list.getEntries().forEach((e) => longTasks.push(e.duration)));
    try {
      observer.observe({ entryTypes: ["longtask"] });
    } catch {
      // long-task timing not supported: reported as 0
    }
    renderProbe.onRender = () => {
      statsRef.current.rowRenders++;
    };

    const perInterval = Math.round((TICK_RATE * INTERVAL_MS) / 1000);
    const start = performance.now();
    await new Promise<void>((resolve) => {
      const timer = setInterval(() => {
        for (let k = 0; k < perInterval; k++) {
          const i = Math.floor(rand() * ROWS);
          prices[i] = Math.max(1, prices[i] + (rand() - 0.5) * 0.4);
          const tick: Tick = { t: items[i].ticker, p: Math.round(prices[i] * 100) / 100, c: 0, cur: "USD", ts: 1 };
          sent++;
          if (selected === "naive") {
            // One render per message, the way an unbatched client handles each WebSocket frame.
            flushSync(() =>
              setNaivePrices((prev) => ({ ...prev, [tick.t]: { price: tick.p, changePct: 0, currency: "USD", ts: 1 } })),
            );
          } else {
            feedRef.current?.([tick]);
          }
        }
        if (performance.now() - start >= DURATION_MS) {
          clearInterval(timer);
          resolve();
        }
      }, INTERVAL_MS);
    });
    await new Promise((r) => setTimeout(r, 400)); // let the last batch render
    observer.disconnect();
    renderProbe.onRender = null;

    const durations = [...statsRef.current.durations].sort((a, b) => a - b);
    const out: BenchResult = {
      mode: selected,
      rows: ROWS,
      ticksSent: sent,
      tickRatePerSecond: TICK_RATE,
      durationMs: Math.round(performance.now() - start),
      commits: durations.length,
      rowRenders: statsRef.current.rowRenders,
      reactRenderMs: Math.round(durations.reduce((a, b) => a + b, 0) * 10) / 10,
      maxCommitMs: Math.round((durations[durations.length - 1] ?? 0) * 100) / 100,
      p95CommitMs: Math.round(percentile(durations, 0.95) * 100) / 100,
      longTasks: longTasks.length,
      longTaskMs: Math.round(longTasks.reduce((a, b) => a + b, 0)),
    };
    setResult(out);
    return out;
  }, []);

  useEffect(() => {
    (window as unknown as { __runBench: typeof run }).__runBench = run;
  }, [run]);

  return (
    <div className="mx-auto max-w-2xl p-5 font-mono text-xs text-text">
      <h1 className="text-lg font-bold">Render benchmark</h1>
      <p className="mt-1 text-muted">
        {ROWS} rows, {TICK_RATE} ticks/s for {DURATION_MS / 1000}s, uniformly random tickers.
      </p>
      <div className="mt-3 flex gap-2">
        {(["naive", "batched", "full"] as const).map((m) => (
          <button key={m} type="button" onClick={() => void run(m)} className="rounded border border-border px-3 py-1.5">
            Run {m}
          </button>
        ))}
      </div>
      {result && <pre data-testid="bench-result" className="mt-3 whitespace-pre-wrap">{JSON.stringify(result, null, 2)}</pre>}
      <div className="mt-4 max-h-64 overflow-hidden">
        {mode === "naive" && <NaiveList prices={naivePrices} onRender={onRender} />}
        {mode === "batched" && storeRef.current && <StoreList store={storeRef.current} memo={false} onRender={onRender} />}
        {mode === "full" && storeRef.current && <StoreList store={storeRef.current} memo onRender={onRender} />}
      </div>
    </div>
  );
}
