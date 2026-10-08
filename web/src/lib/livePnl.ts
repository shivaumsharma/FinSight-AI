// Recomputes a holding's value and profit/loss from a streamed price, so positions move with the market between page
// loads. The server's numbers stay the baseline: summary totals are adjusted by each holding's change rather than rebuilt,
// so the server's rules about which holdings count toward totals are preserved.

import type { LiveQuote } from "./livePrices";
import type { PortfolioHolding, PortfolioSummary } from "./types";

export function applyLiveHolding(h: PortfolioHolding, live: LiveQuote | undefined): PortfolioHolding {
  if (!live || live.price === h.price || !(live.price > 0)) return h;
  const marketValue = h.quantity * live.price;
  const unrealized = marketValue - h.cost_basis;
  let todayPnl = h.today_pnl;
  if (live.changePct !== null && live.changePct > -100) {
    const previousClose = live.price / (1 + live.changePct / 100);
    todayPnl = h.quantity * (live.price - previousClose);
  }
  return {
    ...h,
    price: live.price,
    change_pct: live.changePct ?? h.change_pct,
    market_value: marketValue,
    unrealized_pnl: unrealized,
    unrealized_pnl_pct: h.cost_basis > 0 ? (unrealized / h.cost_basis) * 100 : null,
    today_pnl: todayPnl,
  };
}

// Keeps the same output object for a holding while neither the holding nor its live quote changed, so memoized rows
// skip re-rendering.
export function createLiveHoldingCache() {
  const cache = new Map<string, { h: PortfolioHolding; live: LiveQuote | undefined; out: PortfolioHolding }>();
  return (h: PortfolioHolding, live: LiveQuote | undefined): PortfolioHolding => {
    const hit = cache.get(h.ticker);
    if (hit && hit.h === h && hit.live === live) return hit.out;
    const out = applyLiveHolding(h, live);
    cache.set(h.ticker, { h, live, out });
    return out;
  };
}

// Totals follow the live holdings only when every holding is in the summary's currency. With mixed currencies the
// totals depend on an FX rate the browser does not have, so they are left as the server computed them.
export function applyLiveSummary(
  summary: PortfolioSummary,
  baseHoldings: PortfolioHolding[],
  liveHoldings: PortfolioHolding[],
): PortfolioSummary {
  if (summary.mixed_currency || summary.excluded_from_summary) return summary;
  if (summary.total_market_value === null) return summary;

  let deltaValue = 0;
  let deltaToday = 0;
  let changed = false;
  baseHoldings.forEach((base, i) => {
    const live = liveHoldings[i];
    if (live === base || base.market_value === null || live.market_value === null) return;
    changed = true;
    deltaValue += live.market_value - base.market_value;
    if (base.today_pnl !== null && live.today_pnl !== null) deltaToday += live.today_pnl - base.today_pnl;
  });
  if (!changed) return summary;

  const totalValue = summary.total_market_value + deltaValue;
  const totalPnl = summary.total_unrealized_pnl === null ? null : summary.total_unrealized_pnl + deltaValue;
  const todayPnl = summary.total_today_pnl === null ? null : summary.total_today_pnl + deltaToday;
  const cost = summary.total_cost_basis;
  const previousValue = totalValue - (todayPnl ?? 0);
  return {
    ...summary,
    total_market_value: totalValue,
    total_unrealized_pnl: totalPnl,
    total_unrealized_pnl_pct: totalPnl !== null && cost ? (totalPnl / cost) * 100 : summary.total_unrealized_pnl_pct,
    total_today_pnl: todayPnl,
    total_today_pnl_pct: todayPnl !== null && previousValue > 0 ? (todayPnl / previousValue) * 100 : summary.total_today_pnl_pct,
  };
}
