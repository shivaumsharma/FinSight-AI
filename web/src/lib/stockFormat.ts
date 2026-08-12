// Formatting helpers shared by the stock-detail-page cards
// (StockHeader/PriceStatistics/FundamentalsCard/...). Deliberately
// separate from ReportView.tsx's own fmtMoney/fmtLargeDollar (those are
// module-private there, and this page has different formatting needs --
// e.g. plain volume/share counts with no currency symbol at all).

export function fmtCompactNumber(n: number | null | undefined): string {
  if (n === null || n === undefined) return "N/A";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toLocaleString();
}

export function fmtCompactMoney(n: number | null | undefined, symbol: string): string {
  if (n === null || n === undefined) return "N/A";
  return `${symbol}${fmtCompactNumber(n)}`;
}

export function fmtPrice(n: number | null | undefined, symbol: string): string {
  if (n === null || n === undefined) return "N/A";
  return `${symbol}${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function fmtPercent(n: number | null | undefined, alreadyFraction = false): string {
  if (n === null || n === undefined) return "N/A";
  const pct = alreadyFraction ? n * 100 : n;
  return `${pct.toFixed(2)}%`;
}

export function fmtRatio(n: number | null | undefined, suffix = "x"): string {
  if (n === null || n === undefined) return "N/A";
  return `${n.toFixed(2)}${suffix}`;
}
