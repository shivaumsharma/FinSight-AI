"""
Unit tests for app/reasoning/portfolio_fit.py -- monkeypatches
MarketDataLoader/get_quote/get_usd_conversion_rate directly, no
network. Same fixture/mocking shape as test_similar_stocks.py, which
this module's _get_company_info cache pattern was copied from.
"""

import pytest

from app.reasoning import portfolio_fit as pf


@pytest.fixture(autouse=True)
def _clear_company_info_cache(monkeypatch):
    monkeypatch.setattr(pf, "_company_info_cache", {})


def _holding(ticker, quantity, avg_cost=1.0):
    return {"ticker": ticker, "quantity": quantity, "avg_cost": avg_cost}


def _patch_company_info(monkeypatch, info_by_ticker):
    class _FakeLoader:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_company_info(self):
            return info_by_ticker.get(self.ticker, {})

    monkeypatch.setattr(pf, "MarketDataLoader", _FakeLoader)


def _patch_quotes(monkeypatch, quotes_by_ticker):
    monkeypatch.setattr(pf, "get_quote", lambda ticker: quotes_by_ticker[ticker])


def _patch_usd_rate(monkeypatch, rate=1.0):
    monkeypatch.setattr(pf, "get_usd_conversion_rate", lambda currency: rate)


# ---------------------------------------------------------------- get_portfolio_sector_allocation

def test_allocation_is_value_weighted_not_count_weighted(monkeypatch):
    holdings = [_holding("MSFT", 1), _holding("XOM", 9)]
    _patch_company_info(monkeypatch, {
        "MSFT": {"sector": "Technology"}, "XOM": {"sector": "Energy"},
    })
    _patch_quotes(monkeypatch, {
        "MSFT": {"price": 100.0, "currency": "USD"}, "XOM": {"price": 100.0, "currency": "USD"},
    })
    _patch_usd_rate(monkeypatch)

    allocation = pf.get_portfolio_sector_allocation(holdings)

    assert allocation == {"Technology": 10.0, "Energy": 90.0}


def test_allocation_groups_unknown_sector_separately(monkeypatch):
    holdings = [_holding("MSFT", 1), _holding("ZZZZ", 1)]
    _patch_company_info(monkeypatch, {"MSFT": {"sector": "Technology"}})
    _patch_quotes(monkeypatch, {
        "MSFT": {"price": 100.0, "currency": "USD"}, "ZZZZ": {"price": 100.0, "currency": "USD"},
    })
    _patch_usd_rate(monkeypatch)

    allocation = pf.get_portfolio_sector_allocation(holdings)

    assert allocation == {"Technology": 50.0, "_unknown": 50.0}


def test_allocation_is_empty_for_no_holdings():
    assert pf.get_portfolio_sector_allocation([]) == {}


def test_allocation_skips_a_holding_whose_quote_fails(monkeypatch):
    holdings = [_holding("MSFT", 1), _holding("BROKEN", 1)]
    _patch_company_info(monkeypatch, {"MSFT": {"sector": "Technology"}})

    def _get_quote(ticker):
        if ticker == "BROKEN":
            raise Exception("quote failed")
        return {"price": 100.0, "currency": "USD"}

    monkeypatch.setattr(pf, "get_quote", _get_quote)
    _patch_usd_rate(monkeypatch)

    allocation = pf.get_portfolio_sector_allocation(holdings)

    assert allocation == {"Technology": 100.0}


def test_allocation_skips_a_holding_with_no_known_fx_rate(monkeypatch):
    holdings = [_holding("MSFT", 1)]
    _patch_company_info(monkeypatch, {"MSFT": {"sector": "Technology"}})
    _patch_quotes(monkeypatch, {"MSFT": {"price": 100.0, "currency": "XYZ"}})
    monkeypatch.setattr(pf, "get_usd_conversion_rate", lambda currency: None)

    assert pf.get_portfolio_sector_allocation(holdings) == {}


# ---------------------------------------------------------------- get_portfolio_fit_for_ticker

def test_fit_for_empty_portfolio(monkeypatch):
    _patch_company_info(monkeypatch, {"AAPL": {"sector": "Technology"}})

    result = pf.get_portfolio_fit_for_ticker("AAPL", [])

    assert result["current_allocation_pct"] is None
    assert "no existing portfolio" in result["summary"].lower()


def test_fit_reports_existing_allocation_when_sector_already_held(monkeypatch):
    holdings = [_holding("MSFT", 1)]
    _patch_company_info(monkeypatch, {"MSFT": {"sector": "Technology"}, "AAPL": {"sector": "Technology"}})
    _patch_quotes(monkeypatch, {"MSFT": {"price": 100.0, "currency": "USD"}})
    _patch_usd_rate(monkeypatch)

    result = pf.get_portfolio_fit_for_ticker("AAPL", holdings)

    assert result["sector"] == "Technology"
    assert result["current_allocation_pct"] == 100.0
    assert "already" in result["summary"].lower()


def test_fit_reports_diversification_when_sector_not_held(monkeypatch):
    holdings = [_holding("XOM", 1)]
    _patch_company_info(monkeypatch, {"XOM": {"sector": "Energy"}, "AAPL": {"sector": "Technology"}})
    _patch_quotes(monkeypatch, {"XOM": {"price": 100.0, "currency": "USD"}})
    _patch_usd_rate(monkeypatch)

    result = pf.get_portfolio_fit_for_ticker("AAPL", holdings)

    assert result["current_allocation_pct"] is None
    assert "diversify" in result["summary"].lower()


def test_fit_degrades_cleanly_when_target_sector_unknown(monkeypatch):
    holdings = [_holding("MSFT", 1)]
    _patch_company_info(monkeypatch, {"MSFT": {"sector": "Technology"}, "ZZZZ": {}})

    result = pf.get_portfolio_fit_for_ticker("ZZZZ", holdings)

    assert result["sector"] is None
    assert result["current_allocation_pct"] is None
    assert "unavailable" in result["summary"].lower() or "isn't available" in result["summary"].lower()
