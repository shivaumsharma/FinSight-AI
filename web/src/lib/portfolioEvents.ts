// Portfolio.tsx and OrderTicket.tsx are separate components rendered
// side-by-side on the same page with independent fetch/state -- a
// simulated order placed in OrderTicket updates the backend's
// portfolio_holdings correctly (confirmed: a reload always shows it),
// but Portfolio has no other way to know its own already-fetched data
// just went stale. A plain DOM CustomEvent is the smallest fix that
// doesn't require lifting both components' state up into page.tsx.
export const PORTFOLIO_UPDATED_EVENT = "finsight:portfolio-updated";

export function notifyPortfolioUpdated() {
  window.dispatchEvent(new Event(PORTFOLIO_UPDATED_EVENT));
}
