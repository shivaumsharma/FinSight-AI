import { describe, expect, it } from "vitest";
import {
  formatCompact,
  formatCompactMoney,
  formatNumber,
  formatPercent,
  formatPrice,
  formatQuantity,
  formatRatio,
  formatSignedPrice,
} from "./numberFormat";

const BAD = [null, undefined, NaN, Infinity, -Infinity];

describe("missing and non-finite input", () => {
  it.each(BAD)("every formatter returns N/A for %s", (v) => {
    expect(formatNumber(v)).toBe("N/A");
    expect(formatPrice(v)).toBe("N/A");
    expect(formatSignedPrice(v)).toBe("N/A");
    expect(formatPercent(v)).toBe("N/A");
    expect(formatRatio(v)).toBe("N/A");
    expect(formatQuantity(v)).toBe("N/A");
    expect(formatCompact(v)).toBe("N/A");
    expect(formatCompactMoney(v)).toBe("N/A");
  });

  it("uses the caller's fallback when given", () => {
    expect(formatPercent(null, { fallback: "--" })).toBe("--");
    expect(formatPrice(undefined, "$", "--")).toBe("--");
  });
});

describe("formatPrice", () => {
  it("shows two decimals with grouping", () => {
    expect(formatPrice(1234.5)).toBe("$1,234.50");
    expect(formatPrice(0)).toBe("$0.00");
    expect(formatPrice(189.999)).toBe("$190.00");
  });

  it("puts the sign before the symbol for negatives", () => {
    expect(formatPrice(-5)).toBe("-$5.00");
  });

  it("never prints negative zero", () => {
    expect(formatPrice(-0)).toBe("$0.00");
    expect(formatPrice(-0.001)).not.toMatch(/^-\$0\.00$/);
  });

  it("keeps significant digits for sub-cent prices instead of showing 0.00", () => {
    expect(formatPrice(0.00001234)).toBe("$0.0000123");
    expect(formatPrice(0.005)).toBe("$0.005");
  });

  it("handles very large values", () => {
    expect(formatPrice(1e12)).toBe("$1,000,000,000,000.00");
  });

  it("accepts another currency symbol", () => {
    expect(formatPrice(2500, "₹")).toBe("₹2,500.00");
  });
});

describe("formatSignedPrice", () => {
  it("adds + and - and leaves zero bare", () => {
    expect(formatSignedPrice(1.234)).toBe("+$1.23");
    expect(formatSignedPrice(-1.234)).toBe("-$1.23");
    expect(formatSignedPrice(0)).toBe("$0.00");
    expect(formatSignedPrice(-0)).toBe("$0.00");
  });

  it("treats a change that rounds to zero as zero", () => {
    expect(formatSignedPrice(0.004)).toBe("$0.00");
    expect(formatSignedPrice(-0.004)).toBe("$0.00");
  });
});

describe("formatPercent", () => {
  it("formats plain and fractional input", () => {
    expect(formatPercent(12.345)).toBe("12.35%");
    expect(formatPercent(0.05, { fraction: true })).toBe("5.00%");
    expect(formatPercent(12.3456, { decimals: 1 })).toBe("12.3%");
  });

  it("signs only when asked", () => {
    expect(formatPercent(2.5, { signed: true })).toBe("+2.50%");
    expect(formatPercent(-2.5, { signed: true })).toBe("-2.50%");
    expect(formatPercent(-2.5)).toBe("-2.50%");
  });

  it("never prints -0.00% or +0.00%", () => {
    expect(formatPercent(-0.004)).toBe("0.00%");
    expect(formatPercent(-0.004, { signed: true })).toBe("0.00%");
    expect(formatPercent(0.004, { signed: true })).toBe("0.00%");
    expect(formatPercent(0, { signed: true })).toBe("0.00%");
    expect(formatPercent(-0)).toBe("0.00%");
  });

  it("handles very small and very large percents", () => {
    expect(formatPercent(0.0001, { decimals: 4 })).toBe("0.0001%");
    expect(formatPercent(123456.789)).toBe("123,456.79%");
  });
});

describe("formatCompact", () => {
  it("picks the right unit", () => {
    expect(formatCompact(999)).toBe("999");
    expect(formatCompact(1500)).toBe("1.5K");
    expect(formatCompact(2_500_000)).toBe("2.50M");
    expect(formatCompact(3.2e9)).toBe("3.20B");
    expect(formatCompact(4.1e12)).toBe("4.10T");
  });

  it("rolls a value that rounds up into the next unit", () => {
    expect(formatCompact(999_999)).toBe("1.00M");
    expect(formatCompact(999_999_999)).toBe("1.00B");
  });

  it("keeps the sign and handles zero", () => {
    expect(formatCompact(-2_500_000)).toBe("-2.50M");
    expect(formatCompact(0)).toBe("0");
    expect(formatCompact(-0)).toBe("0");
  });

  it("keeps decimals on small non-integers", () => {
    expect(formatCompact(12.345)).toBe("12.35");
  });
});

describe("formatCompactMoney", () => {
  it("puts the sign before the symbol", () => {
    expect(formatCompactMoney(2_500_000)).toBe("$2.50M");
    expect(formatCompactMoney(-2_500_000)).toBe("-$2.50M");
  });
});

describe("formatQuantity", () => {
  it("drops trailing zeros and keeps fractional shares", () => {
    expect(formatQuantity(10)).toBe("10");
    expect(formatQuantity(0.5)).toBe("0.5");
    expect(formatQuantity(1.123456789)).toBe("1.123457");
    expect(formatQuantity(1234567)).toBe("1,234,567");
    expect(formatQuantity(0)).toBe("0");
  });
});

describe("formatRatio", () => {
  it("formats with suffix", () => {
    expect(formatRatio(21.456)).toBe("21.46x");
    expect(formatRatio(-1.2, "%", 1)).toBe("-1.2%");
  });
});
