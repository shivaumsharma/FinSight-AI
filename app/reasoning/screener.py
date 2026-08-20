"""
screener.py

Stock screener: filter the SAME "Tracked Universe" Top Movers already
scans (app/reasoning/market_movers.py's get_tracked_universe()) by
market cap, P/E, P/B, dividend yield, day change, and volume.

Reuses get_tracked_universe() (the curated-backtest-tickers ∪ global-
watchlist union) rather than deriving a universe of its own -- same
scope, same "Tracked Universe" honesty framing, not a full-market scan.

Fetches its own fundamentals batch rather than reusing market_movers.py's
_fetch_batch_quotes: that function only parses out the 4 fields Top
Movers needs (price, change_pct, currency, name); the screener needs a
dozen more (marketCap, trailingPE, priceToBook, dividendYield,
regularMarketVolume, fiftyTwoWeekHigh/Low, ...) that are ALREADY present
in the exact same Yahoo v7/finance/quote batch response -- confirmed
live, not guessed (every field below was checked against a real response
for AAPL/KO/RELIANCE.NS during development). Reimplementing the same
one-batched-request-per-chunk pattern here (rather than threading a
"which fields do you want" parameter through market_movers.py's tested,
production-proven function) keeps that module's existing behavior and
tests untouched.

Deliberately does NOT filter on ROE, sector, or anything that needs a
full per-ticker statement fetch (app/analysis/alpha_factors.py's ROE
path derives it from balance-sheet/income-statement data, not this
lightweight quote endpoint) -- doing that market-wide would reintroduce
the exact per-ticker throttling problem market_movers.py's own docstring
documents hitting in production and worked around with one batched
request instead.

dividendYield here is Yahoo's already-scaled-to-percent field (e.g. 2.35
for a 2.35% yield) -- NOT trailingAnnualDividendYield, which is the same
number as a fraction (0.0235). Confirmed live against KO (2.35 vs
0.023021583) during development; using the wrong one would silently
under-report every yield filter by 100x.
"""

import time
from typing import Dict, List, Optional

from yfinance.data import YfData

from app.core.company_resolver import get_company_name
from app.reasoning.market_movers import get_tracked_universe

_screener_cache: dict = {}
# Fundamentals (P/E, market cap, ...) move far slower than a live quote --
# a 5-minute TTL is plenty fresh for a screener (unlike Top Movers' 60s,
# which is showing today's price action) and cuts repeat-filter requests
# (a user adjusting one slider) down to one real fetch per 5 minutes.
_SCREENER_CACHE_TTL_SECONDS = 300
_QUOTE_BATCH_SIZE = 50

_yf_data = YfData()

_NUMERIC_FIELDS = {
    "price", "change_pct", "market_cap", "pe_ratio", "forward_pe", "pb_ratio",
    "dividend_yield", "volume", "avg_volume_3m", "fifty_two_week_high", "fifty_two_week_low",
}


def _fetch_batch_fundamentals(tickers: List[str]) -> Dict[str, dict]:
    """{"TICKER": {...}} for every ticker Yahoo returned usable data for.
    Same per-chunk isolation as market_movers._fetch_batch_quotes: one
    failed chunk must not discard another chunk's results, and a ticker
    missing from Yahoo's own response (delisted/invalid) is simply
    absent, not an error. Any individual field Yahoo omits for a given
    ticker (thin coverage, e.g. no dividendYield on a non-payer) comes
    through as None rather than a fabricated 0 -- a screener that reads
    "$0 dividend" for "we don't know" would mislead a yield filter."""
    fundamentals: Dict[str, dict] = {}
    for i in range(0, len(tickers), _QUOTE_BATCH_SIZE):
        chunk = tickers[i : i + _QUOTE_BATCH_SIZE]
        try:
            resp = _yf_data.get(
                url="https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": ",".join(chunk)},
            )
            results = resp.json().get("quoteResponse", {}).get("result", [])
        except Exception:
            continue
        for r in results:
            symbol = r.get("symbol")
            price = r.get("regularMarketPrice")
            if not symbol or price is None:
                continue
            fundamentals[symbol] = {
                "name": r.get("longName") or r.get("shortName") or get_company_name(symbol) or symbol,
                "currency": r.get("currency") or "USD",
                "price": price,
                "change_pct": r.get("regularMarketChangePercent"),
                "market_cap": r.get("marketCap"),
                "pe_ratio": r.get("trailingPE"),
                "forward_pe": r.get("forwardPE"),
                "pb_ratio": r.get("priceToBook"),
                "dividend_yield": r.get("dividendYield"),
                "volume": r.get("regularMarketVolume"),
                "avg_volume_3m": r.get("averageDailyVolume3Month"),
                "fifty_two_week_high": r.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": r.get("fiftyTwoWeekLow"),
            }
    return fundamentals


def _get_universe_fundamentals() -> Dict[str, dict]:
    universe = get_tracked_universe()
    cache_key = tuple(universe)
    cached = _screener_cache.get(cache_key)
    if cached is not None and time.time() - cached[0] < _SCREENER_CACHE_TTL_SECONDS:
        return cached[1]

    fundamentals = _fetch_batch_fundamentals(universe)
    _screener_cache[cache_key] = (time.time(), fundamentals)
    return fundamentals


def _passes_filters(row: dict, filters: Dict[str, Optional[float]]) -> bool:
    """A filter bound is skipped (never excludes) when the row's own
    value for that field is None -- Yahoo not covering a field for a
    thinly-traded ticker must not be conflated with "fails the filter",
    the same None-is-not-zero reasoning _fetch_batch_fundamentals's own
    docstring gives for dividend_yield specifically, applied uniformly
    to every field here."""
    checks = [
        ("market_cap_min", "market_cap", lambda v, b: v >= b),
        ("market_cap_max", "market_cap", lambda v, b: v <= b),
        ("pe_max", "pe_ratio", lambda v, b: v <= b),
        ("pb_max", "pb_ratio", lambda v, b: v <= b),
        ("dividend_yield_min", "dividend_yield", lambda v, b: v >= b),
        ("change_pct_min", "change_pct", lambda v, b: v >= b),
        ("change_pct_max", "change_pct", lambda v, b: v <= b),
        ("volume_min", "volume", lambda v, b: v >= b),
    ]
    for filter_key, field, cmp in checks:
        bound = filters.get(filter_key)
        if bound is None:
            continue
        value = row.get(field)
        if value is None or not cmp(value, bound):
            return False
    return True


def run_screener(
    filters: Dict[str, Optional[float]],
    sort_by: str = "market_cap",
    sort_desc: bool = True,
    limit: int = 50,
) -> dict:
    """Filters + sorts the Tracked Universe's fundamentals snapshot.
    sort_by falling outside the known numeric fields is treated as
    "market_cap" rather than raising -- a screener page is not worth a
    500 over a bad query param, same "degrade, don't error" posture as
    the rest of this app's read-only market endpoints."""
    if sort_by not in _NUMERIC_FIELDS:
        sort_by = "market_cap"

    fundamentals = _get_universe_fundamentals()
    rows = [{"ticker": ticker, **data} for ticker, data in fundamentals.items()]
    matched = [row for row in rows if _passes_filters(row, filters)]

    # None-valued sort field sinks to the end regardless of sort
    # direction -- negating the value (instead of passing reverse=True,
    # which would also flip the None-last half of the key) keeps "sinks
    # to the end" true for both directions in one plain ascending sort.
    def sort_key(row):
        value = row.get(sort_by)
        if value is None:
            return (1, 0.0)
        return (0, -value if sort_desc else value)

    matched.sort(key=sort_key)

    return {
        "results": matched[:limit],
        "total_matched": len(matched),
        "universe_size": len(rows),
    }
