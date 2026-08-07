"""
Tests for Market Movers / Sentiment Gauge:
- db.py's get_all_distinct_watchlist_tickers / get_global_rating_distribution
- app/reasoning/market_movers.py's get_tracked_universe / get_top_movers /
  _fetch_batch_quotes

Never hits real yfinance/network. get_top_movers tests mock
market_movers._fetch_batch_quotes directly (the seam between "which
tickers to rank" and "how their quotes were actually fetched"); a
dedicated _fetch_batch_quotes section below mocks market_movers._yf_data.get
itself, against fake response objects shaped like Yahoo's real
v7/finance/quote JSON (confirmed against the real endpoint during
development, not guessed).
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

def _fake_batch_quote(price, change_pct, currency="USD", name="Some Co"):
    return {"price": price, "change_pct": change_pct, "currency": currency, "name": name}


def test_get_top_movers_ranks_gainers_and_losers(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA", "BBB", "CCC", "DDD"])
    quotes = {
        "AAA": _fake_batch_quote(100.0, 5.0, name="AAA Inc"),
        "BBB": _fake_batch_quote(100.0, -3.0, name="BBB Inc"),
        "CCC": _fake_batch_quote(100.0, 1.0, name="CCC Inc"),
        "DDD": _fake_batch_quote(100.0, -8.0, name="DDD Inc"),
    }
    monkeypatch.setattr(market_movers, "_fetch_batch_quotes", lambda tickers: quotes)

    result = market_movers.get_top_movers(limit=2)
    assert [g["ticker"] for g in result["gainers"]] == ["AAA", "CCC"]
    assert [l["ticker"] for l in result["losers"]] == ["DDD", "BBB"]
    assert result["gainers"][0]["name"] == "AAA Inc"
    assert result["gainers"][0]["currency"] == "USD"


def test_get_top_movers_includes_a_non_usd_tickers_currency(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["BAJFINANCE.NS"])
    quotes = {"BAJFINANCE.NS": _fake_batch_quote(1000.0, 5.0, currency="INR", name="Bajaj Finance Limited")}
    monkeypatch.setattr(market_movers, "_fetch_batch_quotes", lambda tickers: quotes)

    result = market_movers.get_top_movers(limit=5)
    assert result["gainers"][0]["currency"] == "INR"


def test_get_top_movers_gainers_and_losers_never_overlap_on_a_small_universe(monkeypatch):
    # Only 3 tickers but limit=5 -- without the overlap guard the same
    # ticker would appear in both gainers and losers.
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA", "BBB", "CCC"])
    quotes = {
        "AAA": _fake_batch_quote(100.0, 5.0),
        "BBB": _fake_batch_quote(100.0, -3.0),
        "CCC": _fake_batch_quote(100.0, 1.0),
    }
    monkeypatch.setattr(market_movers, "_fetch_batch_quotes", lambda tickers: quotes)

    result = market_movers.get_top_movers(limit=5)
    gainer_tickers = {g["ticker"] for g in result["gainers"]}
    loser_tickers = {l["ticker"] for l in result["losers"]}
    assert not (gainer_tickers & loser_tickers)


def test_get_top_movers_isolates_a_ticker_missing_from_the_batch_response(monkeypatch):
    # Yahoo's batch endpoint just omits a bad/delisted ticker from its
    # response rather than erroring -- confirmed empirically (WBA,
    # delisted in 2025, silently absent from a real batch response).
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA", "BAD"])
    quotes = {"AAA": _fake_batch_quote(100.0, 2.0)}
    monkeypatch.setattr(market_movers, "_fetch_batch_quotes", lambda tickers: quotes)

    result = market_movers.get_top_movers(limit=5)
    all_tickers = {g["ticker"] for g in result["gainers"]} | {l["ticker"] for l in result["losers"]}
    assert all_tickers == {"AAA"}


def test_get_top_movers_uses_the_cache_within_ttl(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA"])
    call_count = {"n": 0}

    def counting_fetch(tickers):
        call_count["n"] += 1
        return {"AAA": _fake_batch_quote(100.0, 1.0)}

    monkeypatch.setattr(market_movers, "_fetch_batch_quotes", counting_fetch)

    market_movers.get_top_movers(limit=5)
    market_movers.get_top_movers(limit=5)
    assert call_count["n"] == 1  # second call served from cache, no re-fetch


def test_get_top_movers_cache_expires_after_ttl(monkeypatch):
    monkeypatch.setattr(market_movers, "get_tracked_universe", lambda: ["AAA"])
    monkeypatch.setattr(market_movers, "_fetch_batch_quotes", lambda tickers: {"AAA": _fake_batch_quote(100.0, 1.0)})

    market_movers.get_top_movers(limit=5)
    # Simulate TTL expiry by rewriting the cached timestamp far in the past.
    cached_result = market_movers._movers_cache[5][1]
    market_movers._movers_cache[5] = (time.time() - market_movers._MOVERS_CACHE_TTL_SECONDS - 1, cached_result)

    call_count = {"n": 0}

    def counting_fetch(tickers):
        call_count["n"] += 1
        return {"AAA": _fake_batch_quote(100.0, 1.0)}

    monkeypatch.setattr(market_movers, "_fetch_batch_quotes", counting_fetch)
    market_movers.get_top_movers(limit=5)
    assert call_count["n"] == 1


# ---------------------------------------------------------------- _fetch_batch_quotes

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _yahoo_result(symbol, price, previous_close, currency="USD", long_name=None, short_name=None):
    r = {"symbol": symbol, "regularMarketPrice": price, "regularMarketPreviousClose": previous_close, "currency": currency}
    if long_name is not None:
        r["longName"] = long_name
    if short_name is not None:
        r["shortName"] = short_name
    return r


def test_fetch_batch_quotes_parses_real_shaped_yahoo_response(monkeypatch):
    payload = {"quoteResponse": {"result": [
        _yahoo_result("AAPL", 312.41, 311.0, long_name="Apple Inc."),
    ]}}
    monkeypatch.setattr(market_movers._yf_data, "get", lambda url, params: _FakeResponse(payload))

    quotes = market_movers._fetch_batch_quotes(["AAPL"])
    assert quotes["AAPL"]["price"] == 312.41
    assert quotes["AAPL"]["change_pct"] == pytest.approx((312.41 - 311.0) / 311.0 * 100)
    assert quotes["AAPL"]["currency"] == "USD"
    assert quotes["AAPL"]["name"] == "Apple Inc."


def test_fetch_batch_quotes_prefers_long_name_falls_back_to_short_name_then_resolver(monkeypatch):
    payload = {"quoteResponse": {"result": [
        _yahoo_result("A", 1.0, 1.0, short_name="A Short"),
        _yahoo_result("B", 1.0, 1.0),
    ]}}
    monkeypatch.setattr(market_movers._yf_data, "get", lambda url, params: _FakeResponse(payload))
    monkeypatch.setattr(market_movers, "get_company_name", lambda t: "B From Resolver" if t == "B" else None)

    quotes = market_movers._fetch_batch_quotes(["A", "B"])
    assert quotes["A"]["name"] == "A Short"
    assert quotes["B"]["name"] == "B From Resolver"


def test_fetch_batch_quotes_skips_a_result_missing_price_or_previous_close(monkeypatch):
    payload = {"quoteResponse": {"result": [
        {"symbol": "NOPRICE", "regularMarketPreviousClose": 100.0, "currency": "USD"},
        {"symbol": "NOPREVCLOSE", "regularMarketPrice": 100.0, "currency": "USD"},
    ]}}
    monkeypatch.setattr(market_movers._yf_data, "get", lambda url, params: _FakeResponse(payload))

    quotes = market_movers._fetch_batch_quotes(["NOPRICE", "NOPREVCLOSE"])
    assert quotes == {}


def test_fetch_batch_quotes_chunks_large_universes(monkeypatch):
    universe = [f"T{i}" for i in range(market_movers._QUOTE_BATCH_SIZE + 5)]
    seen_chunk_sizes = []

    def fake_get(url, params):
        symbols = params["symbols"].split(",")
        seen_chunk_sizes.append(len(symbols))
        results = [_yahoo_result(s, 100.0, 99.0) for s in symbols]
        return _FakeResponse({"quoteResponse": {"result": results}})

    monkeypatch.setattr(market_movers._yf_data, "get", fake_get)
    quotes = market_movers._fetch_batch_quotes(universe)
    assert seen_chunk_sizes == [market_movers._QUOTE_BATCH_SIZE, 5]
    assert len(quotes) == len(universe)


def test_fetch_batch_quotes_isolates_a_failed_chunk(monkeypatch):
    universe = [f"T{i}" for i in range(market_movers._QUOTE_BATCH_SIZE + 5)]

    def flaky_get(url, params):
        symbols = params["symbols"].split(",")
        if len(symbols) == market_movers._QUOTE_BATCH_SIZE:
            raise Exception("network error on first chunk")
        results = [_yahoo_result(s, 100.0, 99.0) for s in symbols]
        return _FakeResponse({"quoteResponse": {"result": results}})

    monkeypatch.setattr(market_movers._yf_data, "get", flaky_get)
    quotes = market_movers._fetch_batch_quotes(universe)
    # The second chunk's 5 tickers must still come through even though
    # the first chunk's request failed entirely.
    assert len(quotes) == 5
