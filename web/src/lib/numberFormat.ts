// The one place numbers become text. Every price, percent, quantity and compact figure on screen goes through here so
// the edge cases (missing, NaN, negative zero, tiny values, unit boundaries) are handled and tested once.
// Locale is fixed to en-US so output does not change with the viewer's browser settings.

export const NA = "N/A";

const LOCALE = "en-US";
const cache = new Map<string, Intl.NumberFormat>();

function nf(options: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = JSON.stringify(options);
  let f = cache.get(key);
  if (!f) {
    f = new Intl.NumberFormat(LOCALE, options);
    cache.set(key, f);
  }
  return f;
}

export function isNum(n: unknown): n is number {
  return typeof n === "number" && Number.isFinite(n);
}

// Grouped decimal. "negative" sign display means a value that rounds to zero prints "0.00", never "-0.00".
export function formatNumber(n: number | null | undefined, decimals = 2, fallback = NA): string {
  if (!isNum(n)) return fallback;
  return nf({ minimumFractionDigits: decimals, maximumFractionDigits: decimals, signDisplay: "negative" }).format(n);
}

function signed(n: number, decimals: number): string {
  return nf({ minimumFractionDigits: decimals, maximumFractionDigits: decimals, signDisplay: "exceptZero" }).format(n);
}

// Sub-cent prices keep three significant digits (0.00001234 -> 0.0000123) instead of collapsing to 0.00.
function priceDigits(abs: number): Intl.NumberFormatOptions {
  return abs > 0 && abs < 0.01
    ? { maximumSignificantDigits: 3 }
    : { minimumFractionDigits: 2, maximumFractionDigits: 2 };
}

export function formatPrice(n: number | null | undefined, symbol = "$", fallback = NA): string {
  if (!isNum(n)) return fallback;
  const text = nf({ ...priceDigits(Math.abs(n)), signDisplay: "negative" }).format(Math.abs(n));
  return `${n < 0 && Number(text.replace(/,/g, "")) !== 0 ? "-" : ""}${symbol}${text}`;
}

// A change in money: "+$1.23", "-$1.23", "$0.00" when it rounds to nothing.
export function formatSignedPrice(n: number | null | undefined, symbol = "$", fallback = NA): string {
  if (!isNum(n)) return fallback;
  const text = formatNumber(Math.abs(n), 2);
  const isZero = Number(text.replace(/,/g, "")) === 0;
  return `${isZero ? "" : n < 0 ? "-" : "+"}${symbol}${text}`;
}

export interface PercentOptions {
  decimals?: number;
  signed?: boolean;
  // true when the input is a fraction (0.05 means 5%).
  fraction?: boolean;
  fallback?: string;
}

export function formatPercent(n: number | null | undefined, opts: PercentOptions = {}): string {
  const { decimals = 2, signed: withSign = false, fraction = false, fallback = NA } = opts;
  if (!isNum(n)) return fallback;
  const pct = fraction ? n * 100 : n;
  return `${withSign ? signed(pct, decimals) : formatNumber(pct, decimals)}%`;
}

export function formatRatio(n: number | null | undefined, suffix = "x", decimals = 2, fallback = NA): string {
  if (!isNum(n)) return fallback;
  return `${formatNumber(n, decimals)}${suffix}`;
}

// Share counts: whole numbers stay whole, fractional shares keep up to six places, never trailing zeros.
export function formatQuantity(n: number | null | undefined, fallback = NA): string {
  if (!isNum(n)) return fallback;
  return nf({ minimumFractionDigits: 0, maximumFractionDigits: 6, signDisplay: "negative" }).format(n);
}

const UNITS: Array<[number, string]> = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];

// 1,234 -> "1.2K", 2.5e9 -> "2.50B". Values that round up to the next unit move into it (999,999 -> "1.00M", not "1000.0K").
export function formatCompact(n: number | null | undefined, fallback = NA): string {
  if (!isNum(n)) return fallback;
  const abs = Math.abs(n);
  for (let i = 0; i < UNITS.length; i++) {
    const [size, unit] = UNITS[i];
    if (abs < size) continue;
    const decimals = unit === "K" ? 1 : 2;
    const scaled = Number((abs / size).toFixed(decimals));
    if (scaled >= 1000 && i > 0) {
      const [nextSize, nextUnit] = UNITS[i - 1];
      return `${n < 0 ? "-" : ""}${(abs / nextSize).toFixed(2)}${nextUnit}`;
    }
    return `${n < 0 ? "-" : ""}${scaled.toFixed(decimals)}${unit}`;
  }
  return formatNumber(n, Number.isInteger(n) ? 0 : 2);
}

export function formatCompactMoney(n: number | null | undefined, symbol = "$", fallback = NA): string {
  if (!isNum(n)) return fallback;
  const text = formatCompact(n);
  return text.startsWith("-") ? `-${symbol}${text.slice(1)}` : `${symbol}${text}`;
}
