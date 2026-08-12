"""
Unit tests for app/reasoning/similar_stocks.py -- monkeypatches
get_tracked_universe/MarketDataLoader/get_quote directly, no network.
"""

import pytest

from app.reasoning import similar_stocks as ss


@pytest.fixture(autouse=True)
def _clear_company_info_cache(monkeypatch):
    monkeypatch.setattr(ss, "_company_info_cache", {})


def _patch_universe(monkeypatch, tickers):
    monkeypatch.setattr(ss, "get_tracked_universe", lambda: tickers)


def _patch_company_info(monkeypatch, info_by_ticker):
    class _FakeLoader:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_company_info(self):
            return info_by_ticker.get(self.ticker, {})

    monkeypatch.setattr(ss, "MarketDataLoader", _FakeLoader)


def _patch_quotes(monkeypatch, quotes_by_ticker):
    monkeypatch.setattr(ss, "get_quote", lambda ticker: quotes_by_ticker[ticker])


def test_find_similar_stocks_returns_empty_when_sector_is_none(monkeypatch):
    _patch_universe(monkeypatch, ["MSFT", "GOOG"])
    assert ss.find_similar_stocks("AAPL", None) == []


def test_find_similar_stocks_matches_same_sector_only(monkeypatch):
    _patch_universe(monkeypatch, ["MSFT", "XOM"])
    _patch_company_info(monkeypatch, {
        "MSFT": {"sector": "Technology", "company_name": "Microsoft Corporation"},
        "XOM": {"sector": "Energy", "company_name": "Exxon Mobil"},
    })
    _patch_quotes(monkeypatch, {"MSFT": {"price": 400.0, "change_pct": 1.0, "currency": "USD"}})

    result = ss.find_similar_stocks("AAPL", "Technology")

    assert len(result) == 1
    assert result[0]["ticker"] == "MSFT"
    assert result[0]["name"] == "Microsoft Corporation"


def test_find_similar_stocks_excludes_the_ticker_itself(monkeypatch):
    _patch_universe(monkeypatch, ["AAPL", "MSFT"])
    _patch_company_info(monkeypatch, {"MSFT": {"sector": "Technology", "company_name": "Microsoft"}})
    _patch_quotes(monkeypatch, {"MSFT": {"price": 400.0, "change_pct": 1.0, "currency": "USD"}})

    result = ss.find_similar_stocks("AAPL", "Technology")

    assert all(r["ticker"] != "AAPL" for r in result)


def test_find_similar_stocks_respects_the_limit(monkeypatch):
    tickers = [f"T{i}" for i in range(10)]
    _patch_universe(monkeypatch, tickers)
    _patch_company_info(monkeypatch, {t: {"sector": "Technology", "company_name": t} for t in tickers})
    _patch_quotes(monkeypatch, {t: {"price": 1.0, "change_pct": 0.0, "currency": "USD"} for t in tickers})

    result = ss.find_similar_stocks("AAPL", "Technology", limit=3)

    assert len(result) == 3


def test_find_similar_stocks_skips_a_candidate_whose_quote_fails(monkeypatch):
    _patch_universe(monkeypatch, ["MSFT", "GOOG"])
    _patch_company_info(monkeypatch, {
        "MSFT": {"sector": "Technology", "company_name": "Microsoft"},
        "GOOG": {"sector": "Technology", "company_name": "Alphabet"},
    })

    def _get_quote(ticker):
        if ticker == "MSFT":
            raise Exception("quote failed")
        return {"price": 150.0, "change_pct": 0.5, "currency": "USD"}

    monkeypatch.setattr(ss, "get_quote", _get_quote)

    result = ss.find_similar_stocks("AAPL", "Technology")

    assert [r["ticker"] for r in result] == ["GOOG"]
