"""
stock_overview.py

Aggregates the per-ticker "stock detail page" header / price-statistics
/ fundamentals / company-info data into one response shape for GET
/v1/stocks/{ticker}/overview (see web/src/app/stock/[ticker]/page.tsx).

Every field here wires a yfinance `.info` key that MarketDataLoader
already has access to but nothing in this app has ever read (day
range, 52-week range, volume, valuation ratios, analyst target price) --
a WIRING gap, not a missing data source, confirmed against a live
yfinance `.info` fetch before writing this. Deliberately does NOT reach
for anything requiring a paid vendor (shareholding pattern, promoter
pledge, industry P/E, etc.) -- see the approved stock-detail-page plan's
"Explicitly out of scope this pass" section; those fields simply don't
exist here or anywhere else in this app.

Reuses get_quote() for the live, cached price/change/currency (the same
45s-TTL cache Watchlist/MarketMovers already share) rather than reading
`.info`'s own price fields, which are known to lag intraday relative to
fast_info.
"""

from app.data.market_data import MarketDataLoader, get_quote


def get_stock_overview(ticker: str) -> dict:
    """Raises TickerNotFoundError (via get_quote()) for a bad/delisted
    ticker -- callers should let this propagate to
    errors.ticker_not_found(), same convention as every other
    ticker-keyed endpoint in main.py. `.info` fetch failures degrade
    every field it feeds to None rather than raising, since a live price
    is the one piece of data this page cannot function without, but a
    missing 52-week-high or P/E is just a blank tile, not a broken page.
    """
    ticker = ticker.upper()
    quote = get_quote(ticker)

    loader = MarketDataLoader(ticker)
    try:
        info = loader.stock.info or {}
    except Exception:
        info = {}

    next_earnings = loader.get_next_earnings_date()

    return {
        "ticker": ticker,
        "price": quote["price"],
        "change_pct": quote["change_pct"],
        "previous_close": quote["previous_close"],
        "currency": quote["currency"],
        "company_name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "business_summary": info.get("longBusinessSummary"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "country": info.get("country"),
        "exchange": info.get("exchange"),
        "next_earnings_date": next_earnings.isoformat() if next_earnings else None,
        "price_statistics": {
            "open": info.get("open") or info.get("regularMarketOpen"),
            "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
            "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume") or info.get("regularMarketVolume"),
            "average_volume": info.get("averageVolume"),
        },
        "fundamentals": {
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "book_value": info.get("bookValue"),
            "debt_to_equity": info.get("debtToEquity"),
            "trailing_eps": info.get("trailingEps"),
            # yfinance renamed pegRatio -> trailingPegRatio at some point;
            # accept either so this doesn't silently go blank on a
            # version bump.
            "peg_ratio": info.get("trailingPegRatio") or info.get("pegRatio"),
        },
        "analyst": {
            "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
        },
    }
