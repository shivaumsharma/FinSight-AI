"""
corporate_actions_feed.py

build_corporate_actions_feed() is the exact body GET /v1/corporate-actions/feed
used to have inline in main.py, extracted so Feature 5's daily briefing
(app/reasoning/daily_briefing.py) can share one implementation with the
REST endpoint instead of a second, drifting copy -- same reasoning
app/reporting/portfolio_summary.py's own module docstring gives for why
build_portfolio_view() was pulled out of main.py the same way.
"""

from app.api import db
from app.core.company_resolver import get_company_name
from app.data.market_data import get_corporate_actions


def build_corporate_actions_feed(user_id: str, scope: str = "all") -> dict:
    # "portfolio" scope is just the self-reported holdings; "all" adds
    # in the watchlist too -- the same two ticker sources Watchlist.tsx
    # and Portfolio.tsx already track separately, just unioned here into
    # one chronological feed rather than shown per-component. A ticker
    # on both lists contributes its events only once (set union, not
    # concatenation).
    portfolio_tickers = {h["ticker"] for h in db.get_portfolio_holdings(user_id)}
    if scope == "portfolio":
        tickers = portfolio_tickers
    else:
        watchlist_tickers = {w["ticker"] for w in db.get_watchlist(user_id)}
        tickers = watchlist_tickers | portfolio_tickers

    events = []
    for ticker in tickers:
        try:
            actions = get_corporate_actions(ticker)
        except Exception:
            # Same per-ticker isolation as everywhere else -- one bad
            # symbol must not blank out the rest of the feed.
            continue
        name = get_company_name(ticker) or ticker
        if actions["next_earnings_date"]:
            events.append({"ticker": ticker, "name": name, "type": "earnings", "date": actions["next_earnings_date"]})
        if actions["next_ex_dividend_date"]:
            events.append({"ticker": ticker, "name": name, "type": "ex_dividend", "date": actions["next_ex_dividend_date"]})

    # ISO date strings sort correctly as plain strings -- soonest
    # upcoming event first, matching what a "what's coming up" feed
    # should lead with.
    events.sort(key=lambda e: e["date"])
    return {"events": events, "scope": scope}
