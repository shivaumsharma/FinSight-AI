import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LivePriceStore, applyTicks, useLivePrices, type LivePrices } from "./livePrices";
import type { PriceStreamOptions, Tick } from "./priceStream";

const tick = (t: string, p: number, c: number | null = 0): Tick => ({ t, p, c, cur: "USD", ts: 1 });

describe("applyTicks", () => {
  it("returns the same object when nothing changed", () => {
    const prev: LivePrices = { AAPL: { price: 100, changePct: 1, currency: "USD", ts: 1 } };
    expect(applyTicks(prev, [tick("AAPL", 100, 1)])).toBe(prev);
    expect(applyTicks(prev, [])).toBe(prev);
  });

  it("keeps the identity of tickers that did not change", () => {
    const prev: LivePrices = {
      AAPL: { price: 100, changePct: 1, currency: "USD", ts: 1 },
      MSFT: { price: 200, changePct: 2, currency: "USD", ts: 1 },
    };
    const next = applyTicks(prev, [tick("AAPL", 101, 1.5)]);
    expect(next).not.toBe(prev);
    expect(next.MSFT).toBe(prev.MSFT);
    expect(next.AAPL).toEqual({ price: 101, changePct: 1.5, currency: "USD", ts: 1 });
  });

  it("uses the last tick when a batch contains several for one ticker", () => {
    expect(applyTicks({}, [tick("AAPL", 1), tick("AAPL", 2), tick("AAPL", 3)]).AAPL.price).toBe(3);
  });

  it("drops invalid prices instead of showing them", () => {
    const next = applyTicks({}, [tick("A", NaN), tick("B", 0), tick("C", -5), tick("D", Infinity), tick("E", 9)]);
    expect(Object.keys(next)).toEqual(["E"]);
  });

  it("treats a missing or non-finite change as null", () => {
    expect(applyTicks({}, [tick("A", 1, null)]).A.changePct).toBeNull();
    expect(applyTicks({}, [tick("A", 1, NaN)]).A.changePct).toBeNull();
  });
});

function makeStore() {
  const created: Array<{
    options: PriceStreamOptions;
    start: ReturnType<typeof vi.fn>;
    stop: ReturnType<typeof vi.fn>;
    setTickers: ReturnType<typeof vi.fn>;
    reconnectNow: ReturnType<typeof vi.fn>;
  }> = [];
  const store = new LivePriceStore({
    flushMs: 100,
    getEndpoint: async () => ({ token: "t", ws_url: "wss://x" }),
    createStream: (options) => {
      const s = { options, start: vi.fn(), stop: vi.fn(), setTickers: vi.fn(), reconnectNow: vi.fn() };
      created.push(s);
      return s;
    },
  });
  return { store, created };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("LivePriceStore", () => {
  it("opens one connection for the union of watched tickers and closes it when the last watcher leaves", () => {
    const { store, created } = makeStore();
    const releaseA = store.watch(["aapl", "MSFT"]);
    const releaseB = store.watch(["AAPL", "NVDA"]);
    expect(created).toHaveLength(1);
    expect(created[0].start).toHaveBeenCalledTimes(1);
    expect(created[0].setTickers).toHaveBeenLastCalledWith(["AAPL", "MSFT", "NVDA"]);

    releaseA();
    expect(created[0].setTickers).toHaveBeenLastCalledWith(["AAPL", "NVDA"]); // AAPL still wanted by B
    releaseA(); // releasing twice is harmless
    releaseB();
    expect(created[0].stop).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().status).toBe("idle");
  });

  it("batches a burst of ticks into one update per flush window", () => {
    const { store, created } = makeStore();
    store.watch(["AAPL", "MSFT"]);
    const listener = vi.fn();
    store.subscribe(listener);

    for (let i = 1; i <= 200; i++) created[0].options.onTicks([tick(i % 2 ? "AAPL" : "MSFT", 100 + i)]);
    expect(listener).not.toHaveBeenCalled(); // nothing yet: still inside the window

    vi.advanceTimersByTime(100);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().prices.AAPL.price).toBe(299); // last odd i is 199 -> 100 + 199
    expect(store.getSnapshot().prices.MSFT.price).toBe(300); // last even i is 200 -> 100 + 200
  });

  it("does not notify when a flush brings no change", () => {
    const { store, created } = makeStore();
    store.watch(["AAPL"]);
    created[0].options.onTicks([tick("AAPL", 100)]);
    vi.advanceTimersByTime(100);
    const listener = vi.fn();
    store.subscribe(listener);
    created[0].options.onTicks([tick("AAPL", 100)]);
    vi.advanceTimersByTime(100);
    expect(listener).not.toHaveBeenCalled();
  });

  it("publishes status changes", () => {
    const { store, created } = makeStore();
    store.watch(["AAPL"]);
    created[0].options.onStatus("reconnecting");
    expect(store.getSnapshot().status).toBe("reconnecting");
  });
});

describe("useLivePrices", () => {
  it("re-renders only when one of its own tickers changes", () => {
    const { store, created } = makeStore();
    let renders = 0;
    const { result } = renderHook(() => {
      renders++;
      return useLivePrices(["AAPL"], store);
    });
    const initial = renders;

    act(() => {
      created[0].options.onTicks([tick("AAPL", 100)]);
      vi.advanceTimersByTime(100);
    });
    expect(result.current.prices.AAPL.price).toBe(100);
    const afterAapl = renders;
    expect(afterAapl).toBeGreaterThan(initial);

    // Another screen watches MSFT; its ticks must not re-render this hook.
    store.watch(["MSFT"]);
    act(() => {
      created[0].options.onTicks([tick("MSFT", 200), tick("MSFT", 201)]);
      vi.advanceTimersByTime(100);
    });
    expect(renders).toBe(afterAapl);
    expect(result.current.prices.MSFT).toBeUndefined();
  });

  it("keeps a quote's object identity until its price changes", () => {
    const { store, created } = makeStore();
    const { result } = renderHook(() => useLivePrices(["AAPL", "MSFT"], store));
    act(() => {
      created[0].options.onTicks([tick("AAPL", 100), tick("MSFT", 200)]);
      vi.advanceTimersByTime(100);
    });
    const msft = result.current.prices.MSFT;
    act(() => {
      created[0].options.onTicks([tick("AAPL", 101)]);
      vi.advanceTimersByTime(100);
    });
    expect(result.current.prices.AAPL.price).toBe(101);
    expect(result.current.prices.MSFT).toBe(msft);
  });

  it("stops watching on unmount", () => {
    const { store, created } = makeStore();
    const { unmount } = renderHook(() => useLivePrices(["AAPL"], store));
    expect(created[0].start).toHaveBeenCalled();
    unmount();
    expect(created[0].stop).toHaveBeenCalled();
  });
});
