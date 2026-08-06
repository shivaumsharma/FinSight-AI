"""
Unit tests for MarketDataLoader.get_next_earnings_date() (app/data/
market_data.py) -- never hits the network; yf.Ticker() construction
itself is lazy (no HTTP call), and `stock.calendar` is monkeypatched
directly to a fake object so this stays fast and deterministic.
"""

from datetime import date, timedelta

from app.data.market_data import MarketDataLoader


class _FakeStock:
    def __init__(self, calendar):
        self.calendar = calendar


def _loader_with_calendar(calendar):
    loader = MarketDataLoader("MSFT")
    loader.stock = _FakeStock(calendar)
    return loader


def test_get_next_earnings_date_returns_the_upcoming_date():
    tomorrow = date.today() + timedelta(days=1)
    loader = _loader_with_calendar({"Earnings Date": [tomorrow]})
    assert loader.get_next_earnings_date() == tomorrow


def test_get_next_earnings_date_returns_today():
    today = date.today()
    loader = _loader_with_calendar({"Earnings Date": [today]})
    assert loader.get_next_earnings_date() == today


def test_get_next_earnings_date_picks_the_earliest_of_multiple_upcoming_dates():
    near = date.today() + timedelta(days=1)
    far = date.today() + timedelta(days=90)
    loader = _loader_with_calendar({"Earnings Date": [far, near]})
    assert loader.get_next_earnings_date() == near


def test_get_next_earnings_date_ignores_past_dates():
    past = date.today() - timedelta(days=5)
    loader = _loader_with_calendar({"Earnings Date": [past]})
    assert loader.get_next_earnings_date() is None


def test_get_next_earnings_date_returns_none_when_calendar_empty():
    loader = _loader_with_calendar({})
    assert loader.get_next_earnings_date() is None


def test_get_next_earnings_date_returns_none_when_calendar_is_none():
    loader = _loader_with_calendar(None)
    assert loader.get_next_earnings_date() is None


def test_get_next_earnings_date_never_raises_on_unexpected_shape():
    # e.g. yfinance returning something other than the expected
    # {"Earnings Date": [...]} dict shape -- must degrade to None, not
    # crash the report over a calendar-format change on yfinance's side.
    loader = _loader_with_calendar("unexpected shape, not a dict")
    assert loader.get_next_earnings_date() is None


# ---------------------------------------------------------------- get_corporate_actions

import pandas as pd
import pytest

from app.data import market_data as md


class _FakeCorporateActionsStock:
    def __init__(self, calendar=None, dividends=None, splits=None):
        self.calendar = calendar
        self.dividends = dividends if dividends is not None else pd.Series(dtype=float)
        self.splits = splits if splits is not None else pd.Series(dtype=float)


@pytest.fixture(autouse=True)
def _clear_corporate_actions_cache(monkeypatch):
    # get_corporate_actions caches by ticker at module scope -- give
    # each test its own fresh dict so results from one test's fake
    # stock can't leak into another's assertions.
    monkeypatch.setattr(md, "_corporate_actions_cache", {})


def test_get_corporate_actions_returns_upcoming_earnings_and_ex_dividend_dates(monkeypatch):
    tomorrow = date.today() + timedelta(days=1)
    stock = _FakeCorporateActionsStock(calendar={"Earnings Date": [tomorrow], "Ex-Dividend Date": tomorrow})
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions("MSFT")
    assert result["next_earnings_date"] == tomorrow.isoformat()
    assert result["next_ex_dividend_date"] == tomorrow.isoformat()


def test_get_corporate_actions_ignores_a_past_ex_dividend_date(monkeypatch):
    past = date.today() - timedelta(days=5)
    stock = _FakeCorporateActionsStock(calendar={"Ex-Dividend Date": past})
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions("MSFT")
    assert result["next_ex_dividend_date"] is None


def test_get_corporate_actions_returns_the_most_recent_dividend_amount(monkeypatch):
    dividends = pd.Series([0.22, 0.24, 0.27], index=pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01"]))
    stock = _FakeCorporateActionsStock(dividends=dividends)
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions("MSFT")
    assert result["last_dividend_amount"] == 0.27


def test_get_corporate_actions_returns_the_most_recent_split(monkeypatch):
    splits = pd.Series([7.0, 4.0], index=pd.to_datetime(["2014-06-09", "2020-08-31"]))
    stock = _FakeCorporateActionsStock(splits=splits)
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions("AAPL")
    assert result["last_split"] == {"date": "2020-08-31", "ratio": 4.0}


def test_get_corporate_actions_returns_all_none_for_a_company_with_no_history(monkeypatch):
    stock = _FakeCorporateActionsStock()
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions("NEWCO")
    assert result == {
        "next_earnings_date": None,
        "next_ex_dividend_date": None,
        "last_dividend_amount": None,
        "last_split": None,
    }


def test_get_corporate_actions_never_raises_on_a_broken_calendar(monkeypatch):
    stock = _FakeCorporateActionsStock(calendar="unexpected shape, not a dict")
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions("MSFT")
    assert result["next_earnings_date"] is None
    assert result["next_ex_dividend_date"] is None


def test_get_corporate_actions_is_cached_within_the_ttl(monkeypatch):
    calls = []

    def make_stock(ticker):
        calls.append(ticker)
        return _FakeCorporateActionsStock()

    monkeypatch.setattr(md.yf, "Ticker", make_stock)

    md.get_corporate_actions("MSFT")
    md.get_corporate_actions("MSFT")
    assert len(calls) == 1
