import { describe, expect, it } from "vitest";
import { applyLiveHolding, applyLiveSummary, createLiveHoldingCache } from "./livePnl";
import type { LiveQuote } from "./livePrices";
import type { PortfolioHolding, PortfolioSummary } from "./types";

const holding = (over: Partial<PortfolioHolding> = {}): PortfolioHolding => ({
  ticker: "AAPL", quantity: 10, avg_cost: 100, buy_date: null, price: 110, change_pct: 10, currency: "USD",
  cost_basis: 1000, market_value: 1100, unrealized_pnl: 100, unrealized_pnl_pct: 10, today_pnl: 100,
  rating: null, added_at: 1, ...over,
});

const quote = (price: number, changePct: number | null = null): LiveQuote => ({ price, changePct, currency: "USD", ts: 1 });

describe("applyLiveHolding", () => {
  it("recomputes value, profit and loss from the streamed price", () => {
    const out = applyLiveHolding(holding(), quote(120));
    expect(out.market_value).toBe(1200);
    expect(out.unrealized_pnl).toBe(200);
    expect(out.unrealized_pnl_pct).toBe(20);
    expect(out.price).toBe(120);
  });

  it("shows a loss as negative", () => {
    const out = applyLiveHolding(holding(), quote(90));
    expect(out.unrealized_pnl).toBe(-100);
    expect(out.unrealized_pnl_pct).toBe(-10);
  });

  it("derives today's P&L from the day's percent change", () => {
    // 120 after a +20% day means the previous close was 100, so 10 shares gained 200 today.
    expect(applyLiveHolding(holding(), quote(120, 20)).today_pnl).toBeCloseTo(200, 10);
    // 90 after a -10% day: previous close 100, 10 shares lost 100.
    expect(applyLiveHolding(holding(), quote(90, -10)).today_pnl).toBeCloseTo(-100, 10);
  });

  it("keeps the server's today's P&L when the stream gives no change figure", () => {
    expect(applyLiveHolding(holding({ today_pnl: 42 }), quote(120, null)).today_pnl).toBe(42);
  });

  it("returns the same object when there is no live quote, no price change, or an invalid price", () => {
    const h = holding();
    expect(applyLiveHolding(h, undefined)).toBe(h);
    expect(applyLiveHolding(h, quote(110))).toBe(h);
    expect(applyLiveHolding(h, quote(0))).toBe(h);
    expect(applyLiveHolding(h, quote(-3))).toBe(h);
  });

  it("handles a zero cost basis without dividing by zero", () => {
    const out = applyLiveHolding(holding({ cost_basis: 0, avg_cost: 0 }), quote(120));
    expect(out.unrealized_pnl_pct).toBeNull();
  });

  it("works for a holding that had no quote at load time", () => {
    const out = applyLiveHolding(holding({ price: null, market_value: null, unrealized_pnl: null, unrealized_pnl_pct: null }), quote(120));
    expect(out.market_value).toBe(1200);
    expect(out.unrealized_pnl).toBe(200);
  });
});

describe("createLiveHoldingCache", () => {
  it("returns the same output object until the holding or its quote changes", () => {
    const apply = createLiveHoldingCache();
    const h = holding();
    const q = quote(120);
    const first = apply(h, q);
    expect(apply(h, q)).toBe(first);
    expect(apply(h, quote(121))).not.toBe(first);
  });
});

const summary = (over: Partial<PortfolioSummary> = {}): PortfolioSummary => ({
  total_market_value: 2300, total_cost_basis: 2000, total_unrealized_pnl: 300, total_unrealized_pnl_pct: 15,
  total_today_pnl: 150, total_today_pnl_pct: 7.0, currency: "USD", mixed_currency: false, excluded_from_summary: false,
  ...over,
} as PortfolioSummary);

describe("applyLiveSummary", () => {
  const a = holding({ ticker: "AAPL", market_value: 1100, today_pnl: 100 });
  const b = holding({ ticker: "MSFT", cost_basis: 1000, market_value: 1200, unrealized_pnl: 200, today_pnl: 50 });

  it("shifts the totals by each holding's change", () => {
    const liveA = applyLiveHolding(a, quote(120, 20)); // value 1100 -> 1200 (+100), today 100 -> 200 (+100)
    const out = applyLiveSummary(summary(), [a, b], [liveA, b]);
    expect(out.total_market_value).toBe(2400);
    expect(out.total_unrealized_pnl).toBe(400);
    expect(out.total_unrealized_pnl_pct).toBe(20); // 400 / 2000
    expect(out.total_today_pnl).toBeCloseTo(250, 10);
    expect(out.total_cost_basis).toBe(2000);
  });

  it("returns the same summary when nothing moved", () => {
    const s = summary();
    expect(applyLiveSummary(s, [a, b], [a, b])).toBe(s);
  });

  it("leaves mixed-currency totals alone because the FX rate is not known here", () => {
    const s = summary({ mixed_currency: true });
    const liveA = applyLiveHolding(a, quote(120));
    expect(applyLiveSummary(s, [a, b], [liveA, b])).toBe(s);
  });

  it("leaves totals alone when a holding is excluded from them", () => {
    const s = summary({ excluded_from_summary: true });
    expect(applyLiveSummary(s, [a, b], [applyLiveHolding(a, quote(120)), b])).toBe(s);
  });

  it("does not touch totals the server did not compute", () => {
    const s = summary({ total_market_value: null });
    expect(applyLiveSummary(s, [a], [applyLiveHolding(a, quote(120))])).toBe(s);
  });

  it("ignores a holding that had no quote when the totals were computed", () => {
    const noQuote = holding({ ticker: "XYZ", price: null, market_value: null, unrealized_pnl: null, today_pnl: null });
    const s = summary();
    expect(applyLiveSummary(s, [noQuote], [applyLiveHolding(noQuote, quote(50))])).toBe(s);
  });
});
