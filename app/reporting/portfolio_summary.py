"""
portfolio_summary.py

build_portfolio_view() is the exact body GET /v1/portfolio used to have
inline in main.py, extracted so the new fast-chat "what's my portfolio
look like" intent (see app/reasoning/chat_router.py) can share one
implementation with the REST endpoint instead of a second, drifting
copy of this aggregation logic.
"""

from app.api import db
from app.data.market_data import get_quote, get_usd_conversion_rate


def build_portfolio_view(user_id: str) -> dict:
    # Each holding displays in its OWN native currency (price/cost_basis/
    # market_value/today_pnl are never converted at the per-holding
    # level) -- but the aggregate summary below needs one common unit
    # to sum across a mixed USD/INR portfolio meaningfully, so summary
    # totals are converted to USD equivalent via a live FX rate
    # (get_usd_conversion_rate). A holding whose currency has no known
    # FX rate is shown correctly on its own row but excluded from the
    # summary totals (mixed_currency_excluded flags this for the UI)
    # rather than silently mixing incompatible units again.
    holdings = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    total_today_pnl = 0.0
    # Denominator for total_today_pnl_pct -- only the market value of
    # holdings that actually contributed a today_pnl (i.e. had a real
    # previous_close), so a holding with a missing quote doesn't skew
    # the percentage by inflating the denominator without a matching
    # numerator contribution.
    market_value_with_today_pnl = 0.0
    any_non_usd = False
    any_excluded_from_summary = False
    for row in db.get_portfolio_holdings(user_id):
        ticker = row["ticker"]
        quantity = row["quantity"]
        avg_cost = row["avg_cost"]
        try:
            quote = get_quote(ticker)
        except Exception:
            # Same per-ticker isolation as the watchlist -- one
            # bad/delisted holding must not 500 the whole portfolio.
            quote = None

        price = quote["price"] if quote else None
        previous_close = quote.get("previous_close") if quote else None
        currency = quote.get("currency", "USD") if quote else "USD"
        if currency != "USD":
            any_non_usd = True
        cost_basis = quantity * avg_cost
        market_value = price * quantity if price is not None else None
        unrealized_pnl = (market_value - cost_basis) if market_value is not None else None
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if unrealized_pnl is not None and cost_basis else None
        today_pnl = quantity * (price - previous_close) if price is not None and previous_close else None

        usd_rate = get_usd_conversion_rate(currency)
        if usd_rate is None:
            any_excluded_from_summary = True
        else:
            if market_value is not None:
                total_market_value += market_value * usd_rate
            total_cost_basis += cost_basis * usd_rate
            if today_pnl is not None:
                total_today_pnl += today_pnl * usd_rate
                market_value_with_today_pnl += market_value * usd_rate

        holdings.append({
            "ticker": ticker,
            "quantity": quantity,
            "avg_cost": avg_cost,
            "buy_date": row.get("buy_date"),
            "price": price,
            "change_pct": quote["change_pct"] if quote else None,
            "currency": currency,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "today_pnl": today_pnl,
            # Live verdict from this ticker's latest completed report --
            # the SAME lookup the Watchlist already uses, so a holding
            # whose research call has since flipped (e.g. to Sell) shows
            # it here too, not just on the Watchlist card.
            "rating": db.get_latest_rating_for_ticker(user_id, ticker),
            "added_at": row["added_at"],
        })

    total_unrealized_pnl = total_market_value - total_cost_basis if holdings else None
    # `is not None`, not a bare truthy check -- confirmed live: a
    # position bought moments ago (fill price == current quote) has a
    # perfectly real, meaningful P&L of exactly 0.0, which is falsy in
    # Python. The old `if total_unrealized_pnl and total_cost_basis`
    # treated that as "no value" and silently produced pnl_pct=None
    # while total_unrealized_pnl itself stayed 0.0 -- a caller checking
    # only "is pnl None" (daily_briefing.py's _portfolio_line) then
    # crashed trying to format the still-None pnl_pct alongside it.
    total_unrealized_pnl_pct = (
        (total_unrealized_pnl / total_cost_basis * 100)
        if total_unrealized_pnl is not None and total_cost_basis
        else None
    )

    return {
        "holdings": holdings,
        "summary": {
            "total_market_value": total_market_value if holdings else None,
            "total_cost_basis": total_cost_basis if holdings else None,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
            "total_today_pnl": total_today_pnl if market_value_with_today_pnl else None,
            # Percentage against YESTERDAY's value of just the holdings
            # that contributed a today_pnl (today's market value minus
            # today's gain) -- not total_market_value, which may include
            # holdings with no previous_close and would otherwise skew
            # the denominator without a matching numerator contribution.
            "total_today_pnl_pct": (total_today_pnl / (market_value_with_today_pnl - total_today_pnl) * 100)
                if market_value_with_today_pnl and (market_value_with_today_pnl - total_today_pnl) else None,
            # Summary totals are always USD-equivalent, regardless of
            # each holding's own native currency -- mixed_currency tells
            # the frontend to label the total "(USD equiv.)" instead of
            # implying every holding actually trades in dollars.
            "currency": "USD",
            "mixed_currency": any_non_usd,
            "excluded_from_summary": any_excluded_from_summary,
        },
    }
