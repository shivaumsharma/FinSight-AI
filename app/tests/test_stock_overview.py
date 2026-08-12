"""
Unit tests for app/data/stock_overview.py -- monkeypatches get_quote()
and MarketDataLoader.stock.info directly (matching test_market_data.py's
_FakeStock convention), no live network.
"""

from datetime import date, timedelta

from app.data import market_data, stock_overview
from app.data.market_data import MarketDataLoader


class _FakeStock:
    def __init__(self, info, calendar=None):
        self.info = info
        self.calendar = calendar


def _patch_loader(monkeypatch, info, calendar=None):
    fake_stock = _FakeStock(info, calendar)
    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, ticker: setattr(self, "ticker", ticker.upper()) or setattr(self, "stock", fake_stock))


def _patch_quote(monkeypatch, quote):
    monkeypatch.setattr(stock_overview, "get_quote", lambda ticker: quote)


def test_get_stock_overview_wires_price_statistics_and_fundamentals(monkeypatch):
    _patch_quote(monkeypatch, {"price": 227.5, "change_pct": 1.2, "previous_close": 224.8, "currency": "USD"})
    _patch_loader(monkeypatch, {
        "longName": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
        "marketCap": 3_400_000_000_000, "longBusinessSummary": "Apple designs...",
        "website": "https://apple.com", "fullTimeEmployees": 164000, "country": "United States",
        "exchange": "NMS", "open": 225.0, "dayHigh": 228.1, "dayLow": 224.9,
        "fiftyTwoWeekHigh": 260.1, "fiftyTwoWeekLow": 164.1, "volume": 50_000_000,
        "averageVolume": 55_000_000, "trailingPE": 34.2, "forwardPE": 30.1,
        "priceToBook": 45.3, "dividendYield": 0.5, "bookValue": 5.1,
        "debtToEquity": 150.2, "trailingEps": 6.6, "trailingPegRatio": 2.1,
        "targetMeanPrice": 240.0, "targetHighPrice": 280.0, "targetLowPrice": 200.0,
        "numberOfAnalystOpinions": 42,
    })

    result = stock_overview.get_stock_overview("aapl")

    assert result["ticker"] == "AAPL"
    assert result["price"] == 227.5
    assert result["company_name"] == "Apple Inc."
    assert result["price_statistics"]["day_high"] == 228.1
    assert result["price_statistics"]["fifty_two_week_low"] == 164.1
    assert result["fundamentals"]["trailing_pe"] == 34.2
    assert result["fundamentals"]["peg_ratio"] == 2.1
    assert result["analyst"]["target_mean_price"] == 240.0
    assert result["analyst"]["number_of_analyst_opinions"] == 42


def test_get_stock_overview_falls_back_to_regular_market_fields(monkeypatch):
    # Some tickers only populate the regularMarket* variants, not the
    # bare open/dayHigh/dayLow/volume keys -- both must be tried.
    _patch_quote(monkeypatch, {"price": 100.0, "change_pct": 0.0, "previous_close": 100.0, "currency": "USD"})
    _patch_loader(monkeypatch, {
        "regularMarketOpen": 99.0, "regularMarketDayHigh": 101.0,
        "regularMarketDayLow": 98.0, "regularMarketVolume": 1_000_000,
        "pegRatio": 1.5,
    })

    result = stock_overview.get_stock_overview("XYZ")

    assert result["price_statistics"]["open"] == 99.0
    assert result["price_statistics"]["day_high"] == 101.0
    assert result["price_statistics"]["volume"] == 1_000_000
    assert result["fundamentals"]["peg_ratio"] == 1.5


def test_get_stock_overview_degrades_to_none_fields_when_info_fetch_fails(monkeypatch):
    _patch_quote(monkeypatch, {"price": 50.0, "change_pct": -1.0, "previous_close": 50.5, "currency": "USD"})

    class _RaisingStock:
        @property
        def info(self):
            raise RuntimeError("yfinance boom")

        calendar = None

    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, ticker: setattr(self, "ticker", ticker.upper()) or setattr(self, "stock", _RaisingStock()))

    result = stock_overview.get_stock_overview("BROKEN")

    assert result["price"] == 50.0
    assert result["company_name"] is None
    assert result["price_statistics"]["day_high"] is None
    assert result["fundamentals"]["trailing_pe"] is None


def test_get_stock_overview_propagates_ticker_not_found(monkeypatch):
    def _raise(ticker):
        raise market_data.TickerNotFoundError(f"No price data found for {ticker}")

    monkeypatch.setattr(stock_overview, "get_quote", _raise)

    try:
        stock_overview.get_stock_overview("NOTREAL")
        assert False, "expected TickerNotFoundError"
    except market_data.TickerNotFoundError:
        pass


def test_get_stock_overview_includes_next_earnings_date(monkeypatch):
    tomorrow = date.today() + timedelta(days=1)
    _patch_quote(monkeypatch, {"price": 10.0, "change_pct": 0.0, "previous_close": 10.0, "currency": "USD"})
    _patch_loader(monkeypatch, {}, calendar={"Earnings Date": [tomorrow]})

    result = stock_overview.get_stock_overview("ABC")

    assert result["next_earnings_date"] == tomorrow.isoformat()
