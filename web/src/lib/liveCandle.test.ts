import { describe, expect, it } from "vitest";
import { applyTickToBar, exchangeDate, type Bar } from "./liveCandle";

// 2026-10-07 is a Wednesday. 15:00 UTC is 11:00 in New York (EDT).
const ts = (iso: string) => Math.floor(new Date(iso).getTime() / 1000);
const bar = (over: Partial<Bar> = {}): Bar => ({ date: "2026-10-07", open: 100, high: 105, low: 98, close: 102, ...over });

describe("exchangeDate", () => {
  it("uses the exchange's calendar day, not UTC's", () => {
    // 01:30 UTC on the 8th is still the evening of the 7th in New York.
    expect(exchangeDate(ts("2026-10-08T01:30:00Z"), "USD")).toBe("2026-10-07");
    // ...but already the 8th in Mumbai (07:00 IST).
    expect(exchangeDate(ts("2026-10-08T01:30:00Z"), "INR")).toBe("2026-10-08");
  });
});

describe("applyTickToBar", () => {
  it("moves the close of today's bar", () => {
    const out = applyTickToBar(bar(), 103, ts("2026-10-07T15:00:00Z"), "USD")!;
    expect(out.isNew).toBe(false);
    expect(out.bar).toEqual({ date: "2026-10-07", open: 100, high: 105, low: 98, close: 103 });
  });

  it("extends the high and the low when price breaks the range", () => {
    expect(applyTickToBar(bar(), 107, ts("2026-10-07T15:00:00Z"), "USD")!.bar.high).toBe(107);
    expect(applyTickToBar(bar(), 95, ts("2026-10-07T15:00:00Z"), "USD")!.bar.low).toBe(95);
  });

  it("never changes the open", () => {
    expect(applyTickToBar(bar(), 110, ts("2026-10-07T15:00:00Z"), "USD")!.bar.open).toBe(100);
  });

  it("returns null when the tick changes nothing", () => {
    expect(applyTickToBar(bar(), 102, ts("2026-10-07T15:00:00Z"), "USD")).toBeNull();
  });

  it("starts a new bar on the next trading day, opening at the previous close", () => {
    const out = applyTickToBar(bar(), 104, ts("2026-10-08T14:00:00Z"), "USD")!;
    expect(out.isNew).toBe(true);
    expect(out.bar).toEqual({ date: "2026-10-08", open: 102, high: 104, low: 102, close: 104 });
  });

  it("does not paint a candle for a weekend, even if the date moved on", () => {
    const friday = bar({ date: "2026-10-09" }); // a Friday
    expect(applyTickToBar(friday, 110, ts("2026-10-10T15:00:00Z"), "USD")).toBeNull(); // Saturday
    expect(applyTickToBar(friday, 110, ts("2026-10-11T15:00:00Z"), "USD")).toBeNull(); // Sunday
  });

  it("does not start a flat bar when the quote just repeats the previous close", () => {
    expect(applyTickToBar(bar(), 102, ts("2026-10-08T14:00:00Z"), "USD")).toBeNull();
  });

  it("ignores a late tick belonging to an earlier day", () => {
    expect(applyTickToBar(bar(), 99, ts("2026-10-06T15:00:00Z"), "USD")).toBeNull();
  });

  it("ignores invalid prices and a missing bar", () => {
    expect(applyTickToBar(undefined, 100, ts("2026-10-07T15:00:00Z"), "USD")).toBeNull();
    expect(applyTickToBar(bar(), NaN, ts("2026-10-07T15:00:00Z"), "USD")).toBeNull();
    expect(applyTickToBar(bar(), 0, ts("2026-10-07T15:00:00Z"), "USD")).toBeNull();
    expect(applyTickToBar(bar(), -1, ts("2026-10-07T15:00:00Z"), "USD")).toBeNull();
  });
});
