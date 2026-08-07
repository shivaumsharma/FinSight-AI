"""
market_movers.py

"Top Movers": Gainers/Losers ranking over a "tracked universe" -- NOT a
full-market scan (this app has no license for that), but a real,
honestly-scoped pool: every ticker from the curated backtest universe
(scripts/backtest_results_curated_asof12mo_exit0mo.json, the same 79
tickers app/reporting/report_data_builder.py's own comments reference)
unioned with every ticker currently on ANY user's watchlist (a global,
not per-user, union -- see db.get_all_distinct_watchlist_tickers()'s
docstring for why). Confirmed via .dockerignore/Dockerfile inspection
that scripts/ (including this json file) ships inside the deployed
Cloud Run container.

Deliberately labeled "Tracked Universe" everywhere this is surfaced
(API response + frontend copy) so it's never mistaken for full-market
coverage.

Quotes come from ONE batched request to Yahoo's v7/finance/quote
endpoint (accepts many comma-separated symbols), NOT a per-ticker loop
over get_quote()/fast_info -- a real production issue, not a
hypothetical: an ~80-ticker ThreadPoolExecutor scan against fast_info
returned as few as 1-3 successes when run from the deployed Cloud Run
service (even after lowering concurrency and adding a retry), while
the exact same ticker list succeeded 78/79 (only WBA, delisted in
2025, genuinely missing) from a non-cloud IP at the same concurrency --
Yahoo's per-request throttling of that endpoint is evidently far
harsher from Cloud Run's egress IP than the fetch pattern itself. A
single request carrying the whole batch reproduced 78/79 successes
reliably even from the same non-cloud test environment; deployed
behavior is confirmed after each deploy, not just assumed to transfer.

Uses yfinance's internal YfData helper for this request -- not fully
public API, but the same session/crumb/cookie machinery yfinance's own
Ticker/fast_info already depends on internally. Confirmed necessary: a
raw, unauthenticated request to this same endpoint gets a 401
(Yahoo requires the crumb token YfData already knows how to obtain),
so this reuses working auth handling rather than reimplementing it.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

from yfinance.data import YfData

from app.api import db
from app.core.company_resolver import get_company_name

BASE_DIR = Path(__file__).resolve().parents[2]
_CURATED_UNIVERSE_FILE = BASE_DIR / "scripts" / "backtest_results_curated_asof12mo_exit0mo.json"

_movers_cache: dict = {}
_MOVERS_CACHE_TTL_SECONDS = 60

# Yahoo's batch quote endpoint comfortably handles far more than this in
# one request -- this app's tracked universe is currently ~80-100
# tickers, well under any practical limit. Chunking defensively guards
# against a URL-length/response-size edge case if the watchlist union
# grows much larger over time, not a limit this app hits today.
_QUOTE_BATCH_SIZE = 50

_yf_data = YfData()


def _load_curated_universe_tickers() -> List[str]:
    """Static file, loaded once per process -- not re-read on every
    request (it never changes at runtime, unlike the watchlist union
    which does)."""
    try:
        with open(_CURATED_UNIVERSE_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError):
        # Missing/corrupt file must degrade to "watchlist tickers only",
        # not take down the whole endpoint.
        return []
    return sorted({r["ticker"] for r in records if r.get("ticker")})


_curated_universe_cache: List[str] = _load_curated_universe_tickers()


def get_tracked_universe() -> List[str]:
    """The curated backtest tickers (module-level, loaded once) unioned
    with the CURRENT global watchlist tickers (queried fresh every call
    -- cheap, and watchlists change as users add/remove tickers)."""
    watchlist_tickers = db.get_all_distinct_watchlist_tickers()
    return sorted(set(_curated_universe_cache) | {t.upper() for t in watchlist_tickers})


def _fetch_batch_quotes(tickers: List[str]) -> Dict[str, dict]:
    """{"TICKER": {"price", "change_pct", "currency", "name"}, ...} for
    every ticker Yahoo actually returned usable data for. A missing/
    invalid/delisted ticker (e.g. WBA) is simply absent from Yahoo's
    own response -- same per-item isolation the rest of this app
    already relies on, just expressed as "absent from the dict" here
    instead of a raised exception."""
    quotes: Dict[str, dict] = {}
    for i in range(0, len(tickers), _QUOTE_BATCH_SIZE):
        chunk = tickers[i : i + _QUOTE_BATCH_SIZE]
        try:
            resp = _yf_data.get(
                url="https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": ",".join(chunk)},
            )
            results = resp.json().get("quoteResponse", {}).get("result", [])
        except Exception:
            # One chunk's request failing must not discard quotes a
            # previous, successful chunk already produced.
            continue
        for r in results:
            symbol = r.get("symbol")
            price = r.get("regularMarketPrice")
            previous_close = r.get("regularMarketPreviousClose")
            if not symbol or price is None or not previous_close:
                continue
            quotes[symbol] = {
                "price": price,
                "change_pct": (price - previous_close) / previous_close * 100,
                "currency": r.get("currency") or "USD",
                "name": r.get("longName") or r.get("shortName") or get_company_name(symbol) or symbol,
            }
    return quotes


def get_top_movers(limit: int = 5) -> dict:
    """{"gainers": [...], "losers": [...]}, each up to `limit` entries,
    sorted by change_pct. Whole-result cached for
    MOVERS_CACHE_TTL_SECONDS, same short-TTL-in-process pattern as
    market_data.py's own quote cache."""
    cached = _movers_cache.get(limit)
    if cached is not None and time.time() - cached[0] < _MOVERS_CACHE_TTL_SECONDS:
        return cached[1]

    universe = get_tracked_universe()
    quotes = _fetch_batch_quotes(universe)
    results = [
        {"ticker": ticker, "name": q["name"], "price": q["price"], "change_pct": q["change_pct"], "currency": q["currency"]}
        for ticker, q in quotes.items()
    ]

    ranked = sorted(results, key=lambda r: r["change_pct"], reverse=True)
    # Guard against gainers/losers overlapping the SAME ticker when the
    # tracked universe has fewer than 2*limit valid quotes (small test
    # fixtures, or a quiet day with lots of failed fetches) -- losers
    # only starts past wherever gainers ended.
    n = len(ranked)
    losers_start = max(limit, n - limit)
    gainers = ranked[:limit]
    losers = list(reversed(ranked[losers_start:]))
    result = {"gainers": gainers, "losers": losers}
    _movers_cache[limit] = (time.time(), result)
    return result
