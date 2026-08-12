"""
similar_stocks.py

Powers the stock-detail-page's "Similar Stocks" section -- a real
same-sector lookup against this app's own tracked universe (the same
curated-backtest + global-watchlist union market_movers.py already
exposes as "Top Movers"), NOT a market-wide "similar stocks" scan. This
app has no peer-multiple/industry data source (see
relative_valuation.py's own docstring) and no comprehensive sector
index to draw from -- scanning beyond the tracked universe would mean
either a slow full-market crawl or fabricating coverage this app
doesn't have. UI must keep labeling this "From your tracked universe",
never implying market-wide coverage.

Sector/name lookups are the one genuinely new network cost here (the
tracked universe's own batch-quote fetch, reused from
market_movers.py, doesn't include sector) -- capped at
MAX_CANDIDATES_SCANNED per call and cached per-ticker (the whole
company_info dict, not just sector, so a later name lookup for the
same candidate doesn't re-fetch) for COMPANY_INFO_CACHE_TTL_SECONDS, so
the cache self-warms across repeated requests instead of re-fetching
the same candidates every time.
"""

import time
from typing import List, Optional

from app.data.market_data import MarketDataLoader, get_quote
from app.reasoning.market_movers import get_tracked_universe

MAX_CANDIDATES_SCANNED = 25
MAX_RESULTS = 7

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


def find_similar_stocks(ticker: str, sector: Optional[str], limit: int = MAX_RESULTS) -> List[dict]:
    """[{"ticker", "name", "price", "change_pct", "currency"}, ...] for
    up to `limit` other tracked-universe tickers sharing `sector` --
    [] if `sector` is None/empty (nothing to match against) or no
    tracked-universe ticker shares it within the scanned window."""
    if not sector:
        return []

    ticker = ticker.upper()
    candidates = [t for t in get_tracked_universe() if t != ticker][:MAX_CANDIDATES_SCANNED]

    matches = []
    for candidate in candidates:
        info = _get_company_info(candidate)
        if info.get("sector") != sector:
            continue
        try:
            quote = get_quote(candidate)
        except Exception:
            continue
        matches.append({
            "ticker": candidate,
            "name": info.get("company_name") or candidate,
            "price": quote["price"],
            "change_pct": quote["change_pct"],
            "currency": quote["currency"],
        })
        if len(matches) >= limit:
            break

    return matches
