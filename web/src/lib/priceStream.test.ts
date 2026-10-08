import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PriceStream, type SocketLike, type StreamStatus, type Tick } from "./priceStream";

class FakeSocket implements SocketLike {
  static all: FakeSocket[] = [];
  readyState = 0;
  sent: string[] = [];
  closed = false;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: unknown }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) {
    FakeSocket.all.push(this);
  }
  send(d: string) {
    this.sent.push(d);
  }
  close() {
    this.closed = true;
  }
  open() {
    this.onopen?.();
  }
  say(msg: unknown) {
    this.onmessage?.({ data: JSON.stringify(msg) });
  }
  drop() {
    this.onclose?.();
  }
}

const tick = (t: string, p: number): Tick => ({ t, p, c: 0, cur: "USD", ts: 1 });

function setup(over: Record<string, unknown> = {}) {
  const statuses: StreamStatus[] = [];
  const ticks: Tick[] = [];
  const getEndpoint = vi.fn(async () => ({ token: "tok", ws_url: "wss://example.test/v1/prices/stream" }));
  const stream = new PriceStream({
    getEndpoint,
    onTicks: (t) => ticks.push(...t),
    onStatus: (s) => statuses.push(s),
    createSocket: (url) => new FakeSocket(url),
    random: () => 1, // no jitter: delay equals the exponential value
    ...over,
  });
  return { stream, statuses, ticks, getEndpoint };
}

const flush = () => vi.advanceTimersByTimeAsync(0);
const last = () => FakeSocket.all[FakeSocket.all.length - 1];

beforeEach(() => {
  vi.useFakeTimers();
  FakeSocket.all = [];
});
afterEach(() => vi.useRealTimers());

describe("PriceStream", () => {
  it("authenticates on open, goes live on hello, then subscribes", async () => {
    const { stream, statuses } = setup();
    stream.setTickers(["AAPL", "MSFT"]);
    stream.start();
    await flush();
    last().open();
    expect(JSON.parse(last().sent[0])).toEqual({ token: "tok" });
    expect(statuses).toEqual(["connecting"]);

    last().say({ type: "hello" });
    expect(statuses).toEqual(["connecting", "live"]);
    expect(JSON.parse(last().sent[1])).toEqual({ type: "subscribe", tickers: ["AAPL", "MSFT"] });
  });

  it("delivers ticks and ignores malformed frames", async () => {
    const { stream, ticks } = setup();
    stream.start();
    await flush();
    last().open();
    last().say({ type: "hello" });
    last().say({ type: "ticks", ticks: [tick("AAPL", 100), tick("MSFT", 200)] });
    last().onmessage?.({ data: "not json" });
    last().onmessage?.({ data: new ArrayBuffer(2) });
    expect(ticks.map((t) => t.t)).toEqual(["AAPL", "MSFT"]);
  });

  it("resends the subscription when the ticker set changes while live", async () => {
    const { stream } = setup();
    stream.start();
    await flush();
    last().open();
    last().say({ type: "hello" });
    stream.setTickers(["NVDA"]);
    expect(JSON.parse(last().sent.at(-1)!)).toEqual({ type: "subscribe", tickers: ["NVDA"] });
  });

  it("reconnects with exponential backoff after a drop and reports reconnecting", async () => {
    const { stream, statuses } = setup();
    stream.start();
    await flush();
    last().open();
    last().say({ type: "hello" });
    const first = last();

    first.drop();
    expect(statuses.at(-1)).toBe("reconnecting");
    await vi.advanceTimersByTimeAsync(999);
    expect(FakeSocket.all).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1); // 1s backoff
    expect(FakeSocket.all).toHaveLength(2);

    last().drop(); // fails before ever saying hello
    await vi.advanceTimersByTimeAsync(1999);
    expect(FakeSocket.all).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1); // 2s backoff
    expect(FakeSocket.all).toHaveLength(3);
  });

  it("caps the backoff and resets it after a successful connection", async () => {
    const { stream } = setup({ maxDelayMs: 4000 });
    stream.start();
    await flush();
    for (let i = 0; i < 5; i++) {
      last().drop();
      await vi.advanceTimersByTimeAsync(4000);
    }
    expect(FakeSocket.all).toHaveLength(6); // never waited longer than the 4s cap
    last().open();
    last().say({ type: "hello" });
    last().drop();
    await vi.advanceTimersByTimeAsync(1000); // back to the 1s base
    expect(FakeSocket.all).toHaveLength(7);
  });

  it("resubscribes the current tickers on the new connection", async () => {
    const { stream } = setup();
    stream.setTickers(["AAPL"]);
    stream.start();
    await flush();
    last().open();
    last().say({ type: "hello" });
    last().drop();
    await vi.advanceTimersByTimeAsync(1000);
    last().open();
    last().say({ type: "hello" });
    expect(JSON.parse(last().sent.at(-1)!)).toEqual({ type: "subscribe", tickers: ["AAPL"] });
  });

  it("treats a silent connection as dropped (watchdog) and reconnects", async () => {
    const { stream, statuses } = setup({ staleAfterMs: 10_000 });
    stream.start();
    await flush();
    const first = last();
    first.open();
    first.say({ type: "hello" });
    await vi.advanceTimersByTimeAsync(9_000);
    first.say({ type: "hb" }); // a heartbeat keeps it alive
    await vi.advanceTimersByTimeAsync(9_000);
    expect(first.closed).toBe(false);
    await vi.advanceTimersByTimeAsync(1_001); // 10s of silence since the heartbeat
    expect(first.closed).toBe(true);
    expect(statuses.at(-1)).toBe("reconnecting");
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeSocket.all).toHaveLength(2);
  });

  it("fetches a fresh token for every attempt, and retries when the token request fails", async () => {
    let calls = 0;
    const getEndpoint = vi.fn(async () => {
      if (++calls === 1) throw new Error("401");
      return { token: `tok${calls}`, ws_url: "wss://example.test" };
    });
    const { stream } = setup({ getEndpoint });
    stream.start();
    await flush();
    expect(FakeSocket.all).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1000);
    expect(FakeSocket.all).toHaveLength(1);
    last().open();
    expect(JSON.parse(last().sent[0])).toEqual({ token: "tok2" });
  });

  it("reports unavailable after repeated failures without ever connecting, and keeps retrying", async () => {
    const { stream, statuses } = setup({ unavailableAfterFailures: 3, maxDelayMs: 1000 });
    stream.start();
    await flush();
    for (let i = 0; i < 4; i++) {
      last().drop();
      await vi.advanceTimersByTimeAsync(1000);
    }
    expect(statuses).toContain("unavailable");
    expect(FakeSocket.all.length).toBeGreaterThan(4);
    last().open();
    last().say({ type: "hello" });
    expect(statuses.at(-1)).toBe("live");
  });

  it("reconnectNow skips the backoff wait", async () => {
    const { stream } = setup();
    stream.start();
    await flush();
    last().open();
    last().say({ type: "hello" });
    last().drop();
    stream.reconnectNow();
    await flush();
    expect(FakeSocket.all).toHaveLength(2);
  });

  it("stop closes the socket, cancels retries and goes idle", async () => {
    const { stream, statuses } = setup();
    stream.start();
    await flush();
    const s = last();
    s.open();
    s.say({ type: "hello" });
    stream.stop();
    expect(s.closed).toBe(true);
    expect(statuses.at(-1)).toBe("idle");
    s.drop(); // late events from the old socket do nothing
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeSocket.all).toHaveLength(1);
  });

  it("ignores events from a socket that has been replaced", async () => {
    const { stream, ticks } = setup();
    stream.start();
    await flush();
    const old = last();
    old.open();
    old.say({ type: "hello" });
    old.drop();
    await vi.advanceTimersByTimeAsync(1000);
    old.say({ type: "ticks", ticks: [tick("AAPL", 1)] });
    expect(ticks).toHaveLength(0);
  });
});
