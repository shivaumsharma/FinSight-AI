"""
Tests for the stock screener (app/reasoning/screener.py).

Never hits real yfinance/network -- mocks screener._yf_data.get against
fake responses shaped like Yahoo's real v7/finance/quote JSON (same
mocking approach as test_market_movers.py's own _fetch_batch_quotes
section), and mocks get_tracked_universe directly for the
filter/sort tests above that seam.
"""

import time

import pytest

from app.reasoning import screener


@pytest.fixture(autouse=True)
def clear_screener_cache():
    screener._screener_cache.clear()
    yield
    screener._screener_cache.clear()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _yahoo_result(symbol, price=100.0, **overrides):
    r = {"symbol": symbol, "regularMarketPrice": price, "currency": "USD", "longName": f"{symbol} Inc"}
    r.update(overrides)
    return r


def _mock_universe_and_quotes(monkeypatch, tickers, results):
    monkeypatch.setattr(screener, "get_tracked_universe", lambda: tickers)
    payload = {"quoteResponse": {"result": results}}
    monkeypatch.setattr(screener._yf_data, "get", lambda url, params: _FakeResponse(payload))


# ---------------------------------------------------------------- _fetch_batch_fundamentals

def test_fetch_batch_fundamentals_parses_real_shaped_yahoo_response(monkeypatch):
    payload = {"quoteResponse": {"result": [
        _yahoo_result(
            "AAPL", price=312.49, marketCap=4560468312064, trailingPE=35.87, forwardPE=32.75,
            priceToBook=42.45, dividendYield=0.34, regularMarketVolume=22912073,
            averageDailyVolume3Month=56628412, fiftyTwoWeekHigh=344.57, fiftyTwoWeekLow=223.78,
            regularMarketChangePercent=-1.37,
        ),
    ]}}
    monkeypatch.setattr(screener._yf_data, "get", lambda url, params: _FakeResponse(payload))

    fundamentals = screener._fetch_batch_fundamentals(["AAPL"])
    row = fundamentals["AAPL"]
    assert row["market_cap"] == 4560468312064
    assert row["pe_ratio"] == 35.87
    assert row["dividend_yield"] == 0.34  # already percent-scaled, not the 0.0034 fraction
    assert row["volume"] == 22912073
    assert row["name"] == "AAPL Inc"


def test_fetch_batch_fundamentals_missing_field_is_none_not_zero(monkeypatch):
    # A ticker Yahoo has thin coverage for (e.g. no dividendYield because
    # it doesn't pay one) must come through as None, not a fabricated 0 --
    # a 0 would wrongly fail (or pass) a dividend_yield_min filter.
    payload = {"quoteResponse": {"result": [_yahoo_result("ZERODIV")]}}
    monkeypatch.setattr(screener._yf_data, "get", lambda url, params: _FakeResponse(payload))

    row = screener._fetch_batch_fundamentals(["ZERODIV"])["ZERODIV"]
    assert row["dividend_yield"] is None
    assert row["pe_ratio"] is None


def test_fetch_batch_fundamentals_skips_result_missing_price(monkeypatch):
    payload = {"quoteResponse": {"result": [{"symbol": "NOPRICE", "currency": "USD"}]}}
    monkeypatch.setattr(screener._yf_data, "get", lambda url, params: _FakeResponse(payload))

    assert screener._fetch_batch_fundamentals(["NOPRICE"]) == {}


def test_fetch_batch_fundamentals_isolates_a_failed_chunk(monkeypatch):
    universe = [f"T{i}" for i in range(screener._QUOTE_BATCH_SIZE + 5)]

    def flaky_get(url, params):
        symbols = params["symbols"].split(",")
        if len(symbols) == screener._QUOTE_BATCH_SIZE:
            raise Exception("network error on first chunk")
        return _FakeResponse({"quoteResponse": {"result": [_yahoo_result(s) for s in symbols]}})

    monkeypatch.setattr(screener._yf_data, "get", flaky_get)
    fundamentals = screener._fetch_batch_fundamentals(universe)
    assert len(fundamentals) == 5


# ---------------------------------------------------------------- run_screener: filtering

def test_run_screener_filters_by_market_cap_min(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["BIG", "SMALL"], [
        _yahoo_result("BIG", marketCap=1_000_000_000_000),
        _yahoo_result("SMALL", marketCap=1_000_000),
    ])
    result = screener.run_screener({"market_cap_min": 1_000_000_000})
    assert [r["ticker"] for r in result["results"]] == ["BIG"]
    assert result["total_matched"] == 1
    assert result["universe_size"] == 2


def test_run_screener_filters_by_pe_max(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["CHEAP", "EXPENSIVE"], [
        _yahoo_result("CHEAP", trailingPE=8.0),
        _yahoo_result("EXPENSIVE", trailingPE=80.0),
    ])
    result = screener.run_screener({"pe_max": 15.0})
    assert [r["ticker"] for r in result["results"]] == ["CHEAP"]


def test_run_screener_a_row_missing_the_filtered_field_never_passes(monkeypatch):
    # None must not silently satisfy a bound in either direction -- an
    # unknown P/E is not "cheap enough" just because there's no value to
    # compare against pe_max.
    _mock_universe_and_quotes(monkeypatch, ["NOPE"], [_yahoo_result("NOPE")])  # no trailingPE key at all
    result = screener.run_screener({"pe_max": 100.0})
    assert result["results"] == []


def test_run_screener_combines_multiple_filters_as_and(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["A", "B"], [
        _yahoo_result("A", marketCap=2_000_000_000, trailingPE=10.0),
        _yahoo_result("B", marketCap=2_000_000_000, trailingPE=50.0),  # fails pe_max
    ])
    result = screener.run_screener({"market_cap_min": 1_000_000_000, "pe_max": 20.0})
    assert [r["ticker"] for r in result["results"]] == ["A"]


def test_run_screener_no_filters_returns_whole_universe(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["A", "B"], [_yahoo_result("A"), _yahoo_result("B")])
    result = screener.run_screener({})
    assert result["total_matched"] == 2


# ---------------------------------------------------------------- run_screener: sorting

def test_run_screener_sorts_descending_by_default(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["LOW", "HIGH", "MID"], [
        _yahoo_result("LOW", marketCap=1),
        _yahoo_result("HIGH", marketCap=3),
        _yahoo_result("MID", marketCap=2),
    ])
    result = screener.run_screener({}, sort_by="market_cap", sort_desc=True)
    assert [r["ticker"] for r in result["results"]] == ["HIGH", "MID", "LOW"]


def test_run_screener_sorts_ascending_when_requested(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["LOW", "HIGH", "MID"], [
        _yahoo_result("LOW", marketCap=1),
        _yahoo_result("HIGH", marketCap=3),
        _yahoo_result("MID", marketCap=2),
    ])
    result = screener.run_screener({}, sort_by="market_cap", sort_desc=False)
    assert [r["ticker"] for r in result["results"]] == ["LOW", "MID", "HIGH"]


def test_run_screener_none_values_sink_to_the_end_regardless_of_direction(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["HASVAL", "NOVAL"], [
        _yahoo_result("HASVAL", trailingPE=10.0),
        _yahoo_result("NOVAL"),  # no trailingPE
    ])
    desc = screener.run_screener({}, sort_by="pe_ratio", sort_desc=True)
    assert [r["ticker"] for r in desc["results"]] == ["HASVAL", "NOVAL"]

    asc = screener.run_screener({}, sort_by="pe_ratio", sort_desc=False)
    assert [r["ticker"] for r in asc["results"]] == ["HASVAL", "NOVAL"]


def test_run_screener_unknown_sort_field_falls_back_to_market_cap(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["LOW", "HIGH"], [
        _yahoo_result("LOW", marketCap=1),
        _yahoo_result("HIGH", marketCap=2),
    ])
    result = screener.run_screener({}, sort_by="not_a_real_field", sort_desc=True)
    assert [r["ticker"] for r in result["results"]] == ["HIGH", "LOW"]


def test_run_screener_respects_limit(monkeypatch):
    _mock_universe_and_quotes(monkeypatch, ["A", "B", "C"], [
        _yahoo_result("A", marketCap=1), _yahoo_result("B", marketCap=2), _yahoo_result("C", marketCap=3),
    ])
    result = screener.run_screener({}, limit=2)
    assert len(result["results"]) == 2
    assert result["total_matched"] == 3  # limit trims the returned page, not the match count


# ---------------------------------------------------------------- caching

def test_run_screener_uses_the_cache_within_ttl(monkeypatch):
    monkeypatch.setattr(screener, "get_tracked_universe", lambda: ["A"])
    call_count = {"n": 0}

    def counting_fetch(tickers):
        call_count["n"] += 1
        return {"A": {"name": "A Inc", "currency": "USD", "price": 1.0, "change_pct": None, "market_cap": 1,
                       "pe_ratio": None, "forward_pe": None, "pb_ratio": None, "dividend_yield": None,
                       "volume": None, "avg_volume_3m": None, "fifty_two_week_high": None, "fifty_two_week_low": None}}

    monkeypatch.setattr(screener, "_fetch_batch_fundamentals", counting_fetch)
    screener.run_screener({})
    screener.run_screener({"pe_max": 999})  # different filters, SAME universe -- must still hit cache
    assert call_count["n"] == 1


def test_run_screener_cache_expires_after_ttl(monkeypatch):
    monkeypatch.setattr(screener, "get_tracked_universe", lambda: ["A"])
    call_count = {"n": 0}

    def counting_fetch(tickers):
        call_count["n"] += 1
        return {"A": {"name": "A Inc", "currency": "USD", "price": 1.0, "change_pct": None, "market_cap": 1,
                       "pe_ratio": None, "forward_pe": None, "pb_ratio": None, "dividend_yield": None,
                       "volume": None, "avg_volume_3m": None, "fifty_two_week_high": None, "fifty_two_week_low": None}}

    monkeypatch.setattr(screener, "_fetch_batch_fundamentals", counting_fetch)
    screener.run_screener({})
    cache_key = ("A",)
    cached_result = screener._screener_cache[cache_key][1]
    screener._screener_cache[cache_key] = (time.time() - screener._SCREENER_CACHE_TTL_SECONDS - 1, cached_result)

    screener.run_screener({})
    assert call_count["n"] == 2
