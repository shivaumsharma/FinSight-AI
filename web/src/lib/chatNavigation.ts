// Where the assistant should take the user after a chat reply, so an
// action (or a question) actually SHOWS its result instead of leaving
// them stuck looking at the chat bubble that describes it -- e.g. "add
// Bajaj Finance to my watchlist" succeeding server-side but never
// visibly landing anywhere confusingly reads as "did that even work?"
// (confirmed live). Deliberately data, not JSX -- easy to reason about
// and unit-test on its own, no rendering involved.
//
// Intents not listed here (advice_request, daily_briefing, general,
// list_alerts) have no single page that represents their answer, so
// they intentionally stay put -- see getNavigationTarget's own comment.
// list_alerts specifically: checked live, this app has no dedicated
// page listing a user's price alerts at all (confirmed by reading the
// only other place "alerts" is even mentioned in the frontend -- a
// notification-permission toggle's own comment, not a list view) -- so
// unlike every other action here, there's nowhere real to send this
// one. Sending it anywhere would just recreate the exact "took me
// somewhere but I still can't see it" complaint this file exists to
// fix, only one level more subtly (a page that LOOKS plausible but
// doesn't actually show the thing).
const TICKER_ROUTE_INTENTS = new Set([
  "ticker_question",
  "portfolio_fit",
  "full_report_request",
  // create_alert names a specific ticker (it's in the backend's own
  // _TICKER_SCOPED_INTENTS) but -- same "no dedicated page" gap as
  // list_alerts above -- there's nowhere that shows the alert itself
  // either. The stock's own page is still real, relevant context for
  // what was just asked about, which is a genuinely better landing
  // spot than staying on the chat bubble; it just doesn't confirm the
  // alert condition visually the way /watchlist confirms a watchlist
  // add.
  "create_alert",
]);

const HOME_INTENTS = new Set(["place_order", "add_holding"]);
const WATCHLIST_INTENTS = new Set(["watchlist_add", "watchlist_remove"]);

// `end_voice_session` is handled separately by the caller (it stops the
// mic, not the router) -- deliberately excluded here so it can never be
// double-handled as an ordinary navigation target too.
export function getNavigationTarget(intent: string | null, ticker: string | null): string | null {
  if (!intent) return null;
  if (TICKER_ROUTE_INTENTS.has(intent)) return ticker ? `/stock/${ticker}` : null;
  if (intent === "stock_discovery_request") return "/screener";
  if (HOME_INTENTS.has(intent)) return "/";
  if (WATCHLIST_INTENTS.has(intent)) return "/watchlist";
  return null;
}
