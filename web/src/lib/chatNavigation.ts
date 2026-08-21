// Where the assistant should take the user after a chat reply, so an
// action (or a question) actually SHOWS its result instead of leaving
// them stuck looking at the chat bubble that describes it -- e.g. "add
// Bajaj Finance to my watchlist" succeeding server-side but never
// visibly landing anywhere confusingly reads as "did that even work?"
// (confirmed live). Deliberately data, not JSX -- easy to reason about
// and unit-test on its own, no rendering involved.
//
// Intents not listed here (advice_request, daily_briefing, general) have
// no single page that represents their answer, so they intentionally
// stay put -- see getNavigationTarget's own comment.
const TICKER_ROUTE_INTENTS = new Set([
  "ticker_question",
  "portfolio_fit",
  "full_report_request",
]);

const HOME_INTENTS = new Set(["place_order", "add_holding"]);
const PROFILE_INTENTS = new Set(["create_alert", "list_alerts"]);
const WATCHLIST_INTENTS = new Set(["watchlist_add", "watchlist_remove"]);

// `end_voice_session` is handled separately by the caller (it stops the
// mic, not the router) -- deliberately excluded here so it can never be
// double-handled as an ordinary navigation target too.
export function getNavigationTarget(intent: string | null, ticker: string | null): string | null {
  if (!intent) return null;
  if (TICKER_ROUTE_INTENTS.has(intent)) return ticker ? `/stock/${ticker}` : null;
  if (intent === "stock_discovery_request") return "/screener";
  if (HOME_INTENTS.has(intent)) return "/";
  if (PROFILE_INTENTS.has(intent)) return "/profile";
  if (WATCHLIST_INTENTS.has(intent)) return "/watchlist";
  return null;
}
