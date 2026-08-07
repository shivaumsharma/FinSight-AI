"""
Tests for Market Movers / Sentiment Gauge:
- db.py's get_all_distinct_watchlist_tickers / get_global_rating_distribution
- app/reasoning/market_movers.py's get_tracked_universe / get_top_movers

Never hits real yfinance/network -- market_movers.py does
`from app.data.market_data import get_quote` and
`from app.core.company_resolver import get_company_name`, so tests
monkeypatch those names directly on the market_movers module (same
"patch where it's looked up, not where it's defined" principle
test_watchlist.py's own docstring states).
"""

import time

import pytest

from app.api import db
from app.reasoning import market_movers


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


@pytest.fixture(autouse=True)
def clear_movers_cache():
    # Module-level _movers_cache persists across tests otherwise --
    # every test here must see a fresh cache, not a previous test's
    # cached result for the same `limit`.
    market_movers._movers_cache.clear()
    yield
    market_movers._movers_cache.clear()


def _complete_job(temp_db, user_id, ticker, rating, started_at=None):
    job_id = temp_db.create_job("q", "hand_rolled", user_id=user_id)
    temp_db.mark_running(job_id)
    if started_at is not None:
        with temp_db._connect() as conn:
            conn.execute("UPDATE jobs SET started_at=? WHERE job_id=?", (started_at, job_id))
    result = {"ticker": ticker, "report_data": {"recommendation": {"rating": rating}}}
    temp_db.mark_done(job_id, result, pdf_path="/tmp/fake.pdf")
    return job_id


# ---------------------------------------------------------------- db: watchlist union

def test_get_all_distinct_watchlist_tickers_dedupes_across_users(temp_db):
    u1 = temp_db.create_user("a@example.com", "h", "s")
    u2 = temp_db.create_user("b@example.com", "h", "s")
    temp_db.add_watchlist_item(u1, "AAPL")
    temp_db.add_watchlist_item(u2, "AAPL")
    temp_db.add_watchlist_item(u2, "MSFT")

    tickers = temp_db.get_all_distinct_watchlist_tickers()
    assert sorted(tickers) == ["AAPL", "MSFT"]


def test_get_all_distinct_watchlist_tickers_empty_when_no_watchlists(temp_db):
    assert temp_db.get_all_distinct_watchlist_tickers() == []


# ---------------------------------------------------------------- db: rating distribution

def test_global_rating_distribution_counts_each_ticker_once_at_latest_rating(temp_db):
    u1 = temp_db.create_user("a@example.com", "h", "s")
    u2 = temp_db.create_user("b@example.com", "h", "s")
    # Same ticker researched by two different users -- only the NEWER
    # of the two ratings should count, not both.
    _complete_job(temp_db, u1, "AAPL", "Sell", started_at=100.0)
    _complete_job(temp_db, u2, "AAPL", "Buy", started_at=200.0)
    _complete_job(temp_db, u1, "MSFT", "Hold", started_at=150.0)

    counts = temp_db.get_global_rating_distribution()
    assert counts == {"Buy": 1, "Hold": 1, "Sell": 0, "Insufficient Data": 0}


def test_global_rating_distribution_empty_when_no_completed_jobs(temp_db):
    counts = temp_db.get_global_rating_distribution()
    assert counts == {"Buy": 0, "Hold": 0, "Sell": 0, "Insufficient Data": 0}


# ---------------------------------------------------------------- tracked universe

def test_get_tracked_universe_unions_curated_and_watchlist_tickers(temp_db, monkeypatch):
    monkeypatch.setattr(market_movers, "_curated_universe_cache", ["INTC", "AAPL"])
    u1 = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(u1, "tsla")  # stored uppercased by add_watchlist_item's caller normally, but exercise raw casing too

    universe = market_movers.get_tracked_universe()
    assert set(universe) >= {"INTC", "AAPL"}
    assert "TSLA" in universe


# ---------------------------------------------------------------- get_top_movers

def _fake_quote(price, change_pct):
    return {"price": price, "change_pct": change_pct, "previous_close": price - change_pct}


def test_get_top_movers_ranks_gainers_and_losers(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA", "BBB", "CCC", "DDD"])
    monkeypatch.setattr(market_movers, "get_company_name", lambda t: f"{t} Inc")

    quotes = {"AAA": 5.0, "BBB": -3.0, "CCC": 1.0, "DDD": -8.0}

    def fake_get_quote(ticker):
        return _fake_quote(100.0, quotes[ticker])

    monkeypatch.setattr(market_movers, "get_quote", fake_get_quote)

    result = market_movers.get_top_movers(limit=2)
    assert [g["ticker"] for g in result["gainers"]] == ["AAA", "CCC"]
    assert [l["ticker"] for l in result["losers"]] == ["DDD", "BBB"]
    assert result["gainers"][0]["name"] == "AAA Inc"


def test_get_top_movers_gainers_and_losers_never_overlap_on_a_small_universe(monkeypatch):
    # Only 3 tickers but limit=5 -- without the overlap guard the same
    # ticker would appear in both gainers and losers.
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr(market_movers, "get_company_name", lambda t: None)
    quotes = {"AAA": 5.0, "BBB": -3.0, "CCC": 1.0}
    monkeypatch.setattr(market_movers, "get_quote", lambda t: _fake_quote(100.0, quotes[t]))

    result = market_movers.get_top_movers(limit=5)
    gainer_tickers = {g["ticker"] for g in result["gainers"]}
    loser_tickers = {l["ticker"] for l in result["losers"]}
    assert not (gainer_tickers & loser_tickers)


def test_get_top_movers_isolates_one_bad_ticker(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA", "BAD"])
    monkeypatch.setattr(market_movers, "get_company_name", lambda t: None)

    def flaky_quote(ticker):
        if ticker == "BAD":
            raise Exception("no data")
        return _fake_quote(100.0, 2.0)

    monkeypatch.setattr(market_movers, "get_quote", flaky_quote)

    result = market_movers.get_top_movers(limit=5)
    all_tickers = {g["ticker"] for g in result["gainers"]} | {l["ticker"] for l in result["losers"]}
    assert all_tickers == {"AAA"}


def test_get_top_movers_uses_the_cache_within_ttl(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA"])
    monkeypatch.setattr(market_movers, "get_company_name", lambda t: None)
    call_count = {"n": 0}

    def counting_quote(ticker):
        call_count["n"] += 1
        return _fake_quote(100.0, 1.0)

    monkeypatch.setattr(market_movers, "get_quote", counting_quote)

    market_movers.get_top_movers(limit=5)
    market_movers.get_top_movers(limit=5)
    assert call_count["n"] == 1  # second call served from cache, no re-fetch


def test_get_top_movers_cache_expires_after_ttl(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA"])
    monkeypatch.setattr(market_movers, "get_company_name", lambda t: None)
    monkeypatch.setattr(market_movers, "get_quote", lambda t: _fake_quote(100.0, 1.0))

    market_movers.get_top_movers(limit=5)
    # Simulate TTL expiry by rewriting the cached timestamp far in the past.
    cached_result = market_movers._movers_cache[5][1]
    market_movers._movers_cache[5] = (time.time() - market_movers._MOVERS_CACHE_TTL_SECONDS - 1, cached_result)

    call_count = {"n": 0}

    def counting_quote(ticker):
        call_count["n"] += 1
        return _fake_quote(100.0, 1.0)

    monkeypatch.setattr(market_movers, "get_quote", counting_quote)
    market_movers.get_top_movers(limit=5)
    assert call_count["n"] == 1
