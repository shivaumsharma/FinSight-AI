"""
portfolio_fit.py

Powers the stock-detail-page's "Portfolio Fit" card -- does a ticker
the user is looking at overlap with sectors they're already exposed
to, or would it diversify them into something new? See GET
/v1/stocks/{ticker}/portfolio-fit in app/api/main.py.

Deliberately LLM-free -- same "algorithmically-derived, not an LLM
narrative" philosophy as stock_score.py. Sector comes from yfinance's
`.info` (via MarketDataLoader.get_company_info()), cached with the
exact same small TTL-cache pattern app/reasoning/similar_stocks.py's
own _get_company_info already uses -- duplicated locally rather than
imported, consistent with how this same pattern already repeats across
stock_score.py/similar_stocks.py/peer_comparison.py in this codebase.

Value-weights each sector's % of the portfolio using the same
USD-conversion math main.py's GET /v1/portfolio/analysis already
computes (get_usd_conversion_rate, quantity * price), not a fresh
independent calculation.
"""

import time
from typing import Optional

from app.data.market_data import MarketDataLoader, get_quote, get_usd_conversion_rate

_company_info_cache: dict = {}
_COMPANY_INFO_CACHE_TTL_SECONDS = 6 * 3600


def _get_company_info(ticker: str) -> dict:
    cached = _company_info_cache.get(ticker)
    if cached is not None and time.time() - cached[0] < _COMPANY_INFO_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        info = MarketDataLoader(ticker).get_company_info()
    except Exception:
        info = {}
    _company_info_cache[ticker] = (time.time(), info)
    return info


def get_portfolio_sector_allocation(holdings: list) -> dict:
    """holdings: db.get_portfolio_holdings()'s own row shape
    ({"ticker", "quantity", "avg_cost", ...}). Returns {sector: pct}
    (0-100, USD-value-weighted) for every sector represented, plus
    "_unknown" for a holding whose sector couldn't be resolved (never
    silently dropped from the denominator, same as unresearched_tickers
    in main.py's get_portfolio_analysis). {} for an empty portfolio."""
    value_by_sector: dict = {}
    total_value = 0.0

    for row in holdings:
        ticker = row["ticker"]
        try:
            quote = get_quote(ticker)
        except Exception:
            continue

        price = quote.get("price")
        if price is None:
            continue
        usd_rate = get_usd_conversion_rate(quote.get("currency", "USD"))
        if usd_rate is None:
            continue

        value_usd = row["quantity"] * price * usd_rate
        sector = _get_company_info(ticker).get("sector") or "_unknown"

        value_by_sector[sector] = value_by_sector.get(sector, 0.0) + value_usd
        total_value += value_usd

    if not total_value:
        return {}

    return {sector: round(value / total_value * 100, 1) for sector, value in value_by_sector.items()}


def get_portfolio_fit_for_ticker(ticker: str, holdings: list) -> dict:
    """{"sector": str | None, "current_allocation_pct": float | None,
    "summary": str} -- current_allocation_pct is this ticker's OWN
    sector's existing weight in the portfolio (None if that sector
    isn't held at all, or the portfolio is empty). Never raises --
    a sector-lookup failure just reads as "sector unknown", same
    degrade-gracefully convention as everywhere else in this app."""
    sector: Optional[str] = _get_company_info(ticker).get("sector")

    if not holdings:
        return {
            "sector": sector,
            "current_allocation_pct": None,
            "summary": "You have no existing portfolio holdings to compare this against yet.",
        }

    if not sector:
        return {
            "sector": None,
            "current_allocation_pct": None,
            "summary": "This ticker's sector isn't available, so it can't be compared against your portfolio's sector mix.",
        }

    allocation = get_portfolio_sector_allocation(holdings)
    pct = allocation.get(sector)

    if pct is None:
        summary = f"This would diversify you into {sector}, a sector you don't currently hold."
    else:
        summary = f"You're already {pct:g}% allocated to {sector}; this ticker is also {sector}."

    return {"sector": sector, "current_allocation_pct": pct, "summary": summary}
