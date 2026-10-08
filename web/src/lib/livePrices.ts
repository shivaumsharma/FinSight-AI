// One shared live-price connection for the whole app. Screens ask for the tickers they show; the store keeps one
// WebSocket for the union, batches incoming ticks so React re-renders at most once per flush window, and keeps the
// object for a ticker unchanged unless its price changed, so memoized rows skip work.

import { useEffect, useRef, useSyncExternalStore } from "react";
import { PriceStream, type PriceStreamOptions, type StreamStatus, type Tick } from "./priceStream";

export interface LiveQuote {
  price: number;
  changePct: number | null;
  currency: string;
  ts: number;
}

export type LivePrices = Record<string, LiveQuote>;

// Returns `prev` itself when nothing changed, and reuses the existing object for every ticker whose price is the same.
export function applyTicks(prev: LivePrices, ticks: Tick[]): LivePrices {
  let next: LivePrices | null = null;
  for (const tick of ticks) {
    if (!Number.isFinite(tick.p) || tick.p <= 0) continue;
    const current = (next ?? prev)[tick.t];
    const changePct = typeof tick.c === "number" && Number.isFinite(tick.c) ? tick.c : null;
    if (current && current.price === tick.p && current.changePct === changePct) continue;
    if (!next) next = { ...prev };
    next[tick.t] = { price: tick.p, changePct, currency: tick.cur || "USD", ts: tick.ts };
  }
  return next ?? prev;
}

export interface StoreSnapshot {
  prices: LivePrices;
  status: StreamStatus;
}

export interface LivePriceStoreOptions {
  flushMs?: number;
  createStream?: (options: PriceStreamOptions) => Pick<PriceStream, "start" | "stop" | "setTickers" | "reconnectNow">;
  getEndpoint?: PriceStreamOptions["getEndpoint"];
}

async function defaultGetEndpoint() {
  const resp = await fetch("/api/prices/stream-token", { cache: "no-store" });
  if (!resp.ok) throw new Error(`stream token ${resp.status}`);
  return resp.json();
}

export class LivePriceStore {
  private snapshot: StoreSnapshot = { prices: {}, status: "idle" };
  private listeners = new Set<() => void>();
  private refs = new Map<string, number>();
  private pending: Tick[] = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private stream: ReturnType<NonNullable<LivePriceStoreOptions["createStream"]>> | null = null;
  private flushMs: number;
  private createStream: NonNullable<LivePriceStoreOptions["createStream"]>;
  private getEndpoint: PriceStreamOptions["getEndpoint"];

  constructor(options: LivePriceStoreOptions = {}) {
    this.flushMs = options.flushMs ?? 100;
    this.createStream = options.createStream ?? ((o) => new PriceStream(o));
    this.getEndpoint = options.getEndpoint ?? defaultGetEndpoint;
  }

  getSnapshot = (): StoreSnapshot => this.snapshot;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  // Registers interest in tickers; returns the release function. The connection opens with the first interest and
  // closes when the last is released.
  watch(tickers: string[]): () => void {
    const unique = Array.from(new Set(tickers.map((t) => t.toUpperCase())));
    for (const t of unique) this.refs.set(t, (this.refs.get(t) ?? 0) + 1);
    this.sync();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      for (const t of unique) {
        const n = (this.refs.get(t) ?? 1) - 1;
        if (n <= 0) this.refs.delete(t);
        else this.refs.set(t, n);
      }
      this.sync();
    };
  }

  private sync(): void {
    const wanted = Array.from(this.refs.keys()).sort();
    if (wanted.length === 0) {
      this.stream?.stop();
      this.stream = null;
      this.setSnapshot({ ...this.snapshot, status: "idle" });
      return;
    }
    if (!this.stream) {
      this.stream = this.createStream({
        getEndpoint: this.getEndpoint,
        onTicks: (ticks) => this.enqueue(ticks),
        onStatus: (status) => this.setSnapshot({ ...this.snapshot, status }),
      });
      this.stream.setTickers(wanted);
      this.stream.start();
    } else {
      this.stream.setTickers(wanted);
    }
  }

  reconnectNow(): void {
    this.stream?.reconnectNow();
  }

  private enqueue(ticks: Tick[]): void {
    this.pending.push(...ticks);
    if (this.flushTimer === null) this.flushTimer = setTimeout(() => this.flush(), this.flushMs);
  }

  flush(): void {
    if (this.flushTimer !== null) clearTimeout(this.flushTimer);
    this.flushTimer = null;
    const batch = this.pending;
    this.pending = [];
    const prices = applyTicks(this.snapshot.prices, batch);
    if (prices !== this.snapshot.prices) this.setSnapshot({ ...this.snapshot, prices });
  }

  private setSnapshot(next: StoreSnapshot): void {
    this.snapshot = next;
    this.listeners.forEach((l) => l());
  }
}

let shared: LivePriceStore | null = null;
export function getLivePriceStore(): LivePriceStore {
  if (!shared) shared = new LivePriceStore();
  return shared;
}

const EMPTY: LivePrices = {};

// Live quotes for `tickers` (and the connection status). The returned object only changes when one of these
// tickers' prices changes, not when some other screen's ticker ticks.
export function useLivePrices(tickers: string[], store: LivePriceStore = getLivePriceStore()): StoreSnapshot {
  const key = Array.from(new Set(tickers.map((t) => t.toUpperCase()))).sort().join(",");

  useEffect(() => {
    if (!key) return;
    const release = store.watch(key.split(","));
    const onOnline = () => store.reconnectNow();
    window.addEventListener("online", onOnline);
    return () => {
      window.removeEventListener("online", onOnline);
      release();
    };
  }, [key, store]);

  const last = useRef<StoreSnapshot>({ prices: EMPTY, status: "idle" });
  const getSnapshot = () => {
    const snap = store.getSnapshot();
    const wanted = key ? key.split(",") : [];
    const prev = last.current;
    let same = snap.status === prev.status;
    const subset: LivePrices = {};
    for (const t of wanted) {
      const q = snap.prices[t];
      if (q) subset[t] = q;
      if (same && prev.prices[t] !== q) same = false;
    }
    if (same && Object.keys(subset).length === Object.keys(prev.prices).length) return prev;
    last.current = { prices: subset, status: snap.status };
    return last.current;
  };
  return useSyncExternalStore(store.subscribe, getSnapshot, () => last.current);
}
