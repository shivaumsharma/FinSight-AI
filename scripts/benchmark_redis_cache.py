"""
benchmark_redis_cache.py

Before/after latency on a cache miss vs. a cache hit for the two Redis
integration points that matter most (app/core/cache.py):
  1. ValuationPipeline.run_valuation() -- CPU-bound (WACC, FCFF forecast,
     5x5 sensitivity grid, 2000-sample Monte Carlo), no network.
  2. MarketDataLoader's statement fetches -- network-bound (yfinance).

No real Redis server is required or assumed to be running anywhere
this project deploys to by default (local dev, HF Spaces) -- this uses
fakeredis, a standard PyPI package that implements the actual redis-py
wire protocol in-process, so app/core/cache.py's real cache_get/
cache_set code path is genuinely exercised (not bypassed or mocked),
just against an in-memory stand-in instead of a real Redis server. A
real deployment would set REDIS_URL to a real instance (e.g. Upstash
for HF Spaces, which has no way to run a local Redis daemon) and see
the same behavior against real infrastructure.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fakeredis

from app.core import cache as cache_module
from app.data.market_data import MarketDataLoader
from app.valuation.valuation_pipeline import ValuationPipeline
from app.data.financial_normalizer import FinancialStatementNormaliser

VALUATION_TICKER = "AAPL"
STATEMENT_TICKER = "MSFT"


def _install_fake_redis():
    fake_client = fakeredis.FakeStrictRedis()
    cache_module._client = fake_client
    cache_module._available = True
    cache_module._last_check_time = float("inf")  # never re-check/reconnect
    return fake_client


def _timed(fn):
    start = time.time()
    result = fn()
    return result, time.time() - start


def benchmark_valuation_cache():
    print(f"\n{'=' * 70}\n1. ValuationPipeline.run_valuation() -- {VALUATION_TICKER}\n{'=' * 70}")

    loader = MarketDataLoader(VALUATION_TICKER)
    normaliser = FinancialStatementNormaliser(
        loader.get_income_statement(), loader.get_balance_sheet(), loader.get_cash_flow(),
    )
    financial_df = normaliser.normalise()
    company_info = loader.get_company_info()

    def run():
        return ValuationPipeline(
            financial_df=financial_df,
            market_cap=company_info.get("market_cap"),
            beta=company_info.get("beta") or 1.2,
            ticker=VALUATION_TICKER,
        ).run_valuation()

    _, t_miss = _timed(run)   # first call: cache miss, real computation
    _, t_hit = _timed(run)    # second call: cache hit

    print(f"  Cache miss (real computation): {t_miss * 1000:.2f}ms")
    print(f"  Cache hit:                     {t_hit * 1000:.2f}ms")
    print(f"  Speedup: {t_miss / t_hit if t_hit else float('inf'):.1f}x")
    return t_miss, t_hit


def benchmark_statement_cache(ticker: str):
    print(f"\n{'=' * 70}\n2. MarketDataLoader statement fetches -- {ticker}\n{'=' * 70}")

    # A DIFFERENT ticker than benchmark_valuation_cache() above, not
    # just a fresh MarketDataLoader/yf.Ticker instance -- the Redis
    # cache itself is keyed by ticker, so reusing AAPL here would make
    # the "miss" timing a hit against the cache already warmed by the
    # valuation benchmark's own statement fetches, not a genuine miss.
    loader = MarketDataLoader(ticker)

    def fetch_all():
        loader.get_income_statement()
        loader.get_balance_sheet()
        loader.get_cash_flow()

    _, t_miss = _timed(fetch_all)  # first call: real yfinance network fetches
    _, t_hit = _timed(fetch_all)   # second call: cached

    print(f"  Cache miss (3 real yfinance calls): {t_miss * 1000:.2f}ms")
    print(f"  Cache hit (3 cached lookups):       {t_hit * 1000:.2f}ms")
    print(f"  Speedup: {t_miss / t_hit if t_hit else float('inf'):.1f}x")
    return t_miss, t_hit


def main():
    _install_fake_redis()
    print("Using fakeredis (in-process, real redis-py wire protocol) -- see module docstring.")

    val_miss, val_hit = benchmark_valuation_cache()
    stmt_miss, stmt_hit = benchmark_statement_cache(STATEMENT_TICKER)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"Valuation:  {val_miss*1000:.0f}ms -> {val_hit*1000:.2f}ms  ({val_miss/val_hit:.1f}x)")
    print(f"Statements: {stmt_miss*1000:.0f}ms -> {stmt_hit*1000:.2f}ms  ({stmt_miss/stmt_hit:.1f}x)")
    print(
        "\nBoth are real speedups on their own terms (CPU-bound DCF/Monte Carlo math for "
        "valuation; yfinance network round-trips for statements) -- neither is the dominant "
        "cost in a full report (that's the ~65s LLM narrative call, also now cached -- see "
        "narrative_builder.py -- with the biggest win of all: a full repeat query for the "
        "same evidence, once warm, skips essentially the entire pipeline)."
    )


if __name__ == "__main__":
    main()
