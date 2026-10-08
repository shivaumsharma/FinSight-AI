// Folds a streamed price into the last daily candle so the chart's final bar moves with the market.

export interface Bar {
  date: string; // YYYY-MM-DD, exchange-local
  open: number;
  high: number;
  low: number;
  close: number;
}

const TIME_ZONE_BY_CURRENCY: Record<string, string> = { USD: "America/New_York", INR: "Asia/Kolkata" };

// The trading date a tick belongs to, in the exchange's own time zone (a US tick at 9pm ET is still that day's).
export function exchangeDate(tsSeconds: number, currency: string): string {
  const timeZone = TIME_ZONE_BY_CURRENCY[currency] ?? "UTC";
  return new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit" }).format(
    new Date(tsSeconds * 1000),
  );
}

function isWeekday(isoDate: string): boolean {
  const day = new Date(`${isoDate}T12:00:00Z`).getUTCDay();
  return day !== 0 && day !== 6;
}

// Returns the bar to draw (replacing the last bar, or appended after it), or null when the tick changes nothing.
// A new session's bar is only started on a weekday and only when the price differs from the last close: outside
// trading hours the quote just repeats the previous close and must not paint a flat candle for a weekend or holiday.
export function applyTickToBar(
  last: Bar | undefined,
  price: number,
  tsSeconds: number,
  currency: string,
): { bar: Bar; isNew: boolean } | null {
  if (!last || !Number.isFinite(price) || price <= 0) return null;
  const date = exchangeDate(tsSeconds, currency);

  if (date === last.date) {
    const bar = { ...last, close: price, high: Math.max(last.high, price), low: Math.min(last.low, price) };
    return bar.close === last.close && bar.high === last.high && bar.low === last.low ? null : { bar, isNew: false };
  }
  if (date > last.date && isWeekday(date) && price !== last.close) {
    return { bar: { date, open: last.close, high: Math.max(last.close, price), low: Math.min(last.close, price), close: price }, isNew: true };
  }
  return null;
}
