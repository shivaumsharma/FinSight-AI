// Mirrors app/core/currency.py's CURRENCY_SYMBOLS exactly -- the same
// two-currency scope this app's actual company/index coverage needs
// (US via SEC, India via NSE). Falls back to the raw currency code
// (e.g. "EUR") rather than defaulting to "$" for anything unrecognized
// -- silently mislabeling an unknown currency as dollars is worse than
// showing the plain code.
const CURRENCY_SYMBOLS: Record<string, string> = { USD: "$", INR: "₹" };

export function currencySymbol(currency: string | null | undefined): string {
  if (!currency) return CURRENCY_SYMBOLS.USD;
  return CURRENCY_SYMBOLS[currency] ?? `${currency} `;
}
