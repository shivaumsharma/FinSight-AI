export function relativeTime(unixSeconds: number): string {
  const diffMs = Date.now() - unixSeconds * 1000;
  const minutes = Math.round(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

// For a "YYYY-MM-DD" date string (e.g. corporate-actions dates, news
// article dates) rather than a unix timestamp -- relativeTime above
// doesn't apply since there's no time-of-day component.
export function formatShortDate(isoDate: string): string {
  return new Date(isoDate).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

// Compact large numbers for the Screener's market-cap/volume columns
// (e.g. 4_560_468_312_064 -> "4.56T"). Plain suffix scaling, not
// Intl.NumberFormat's "compact" notation -- that formatter picks
// thousand-groupings by locale (India's lakh/crore vs. the rest of the
// world's thousand/million), which would silently change the unit
// underneath a fixed threshold like a market-cap filter's own input.
export function formatCompact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(0);
}
