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
