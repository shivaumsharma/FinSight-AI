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
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from app.api import db
from app.core.company_resolver import get_company_name
from app.data.market_data import get_quote

BASE_DIR = Path(__file__).resolve().parents[2]
_CURATED_UNIVERSE_FILE = BASE_DIR / "scripts" / "backtest_results_curated_asof12mo_exit0mo.json"

_movers_cache: dict = {}
_MOVERS_CACHE_TTL_SECONDS = 60


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


def _fetch_one(ticker: str) -> dict | None:
    try:
        quote = get_quote(ticker)
    except Exception:
        # Same per-item isolation as every other quote-scan in this app
        # (watchlist, indices, portfolio) -- one bad/delisted ticker
        # must not blank out the whole ranking.
        return None
    if quote["change_pct"] is None:
        return None
    return {
        "ticker": ticker,
        "name": get_company_name(ticker) or ticker,
        "price": quote["price"],
        "change_pct": quote["change_pct"],
    }


def get_top_movers(limit: int = 5) -> dict:
    """{"gainers": [...], "losers": [...]}, each up to `limit` entries,
    sorted by change_pct. Concurrent fetch (ThreadPoolExecutor) since
    the tracked universe can be ~80+ tickers and get_quote() is a
    network call per symbol -- sequential would make this endpoint
    noticeably slow. Whole-result cached for MOVERS_CACHE_TTL_SECONDS,
    same short-TTL-in-process pattern as market_data.py's own quote
    cache (which this still benefits from underneath on a cache miss
    here, since get_quote() has its own 45s cache too)."""
    cached = _movers_cache.get(limit)
    if cached is not None and time.time() - cached[0] < _MOVERS_CACHE_TTL_SECONDS:
        return cached[1]

    universe = get_tracked_universe()
    with ThreadPoolExecutor(max_workers=min(20, max(1, len(universe)))) as pool:
        results = [r for r in pool.map(_fetch_one, universe) if r is not None]

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
