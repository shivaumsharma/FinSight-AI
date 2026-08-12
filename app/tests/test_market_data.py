"""
Unit tests for MarketDataLoader.get_next_earnings_date() (app/data/
market_data.py) -- never hits the network; yf.Ticker() construction
itself is lazy (no HTTP call), and `stock.calendar` is monkeypatched
directly to a fake object so this stays fast and deterministic.
"""

from datetime import date, timedelta

from app.data import market_data
from app.data.market_data import MarketDataLoader, get_usd_conversion_rate


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


# ---------------------------------------------------------------- currency conversion

def test_usd_conversion_rate_is_1_for_usd():
    assert get_usd_conversion_rate("USD") == 1.0


def test_usd_conversion_rate_is_1_for_missing_currency():
    # A holding whose quote never resolved a currency defaults to USD
    # everywhere this app reads it -- same "no conversion needed" case.
    assert get_usd_conversion_rate(None) == 1.0
    assert get_usd_conversion_rate("") == 1.0


def test_usd_conversion_rate_for_inr_is_the_reciprocal_of_the_fx_quote(monkeypatch):
    # INR=X is quoted as INR-per-1-USD (~95), so converting an INR
    # amount TO USD is dividing by that -- i.e. multiplying by its
    # reciprocal.
    monkeypatch.setattr(market_data, "get_quote", lambda ticker: {"price": 95.0, "change_pct": 0.1, "previous_close": 94.9, "currency": "INR"})
    rate = get_usd_conversion_rate("INR")
    assert rate == 1.0 / 95.0


def test_usd_conversion_rate_is_none_for_an_unknown_currency():
    # This app only has a known FX ticker for INR (see
    # _FX_TICKERS_BY_CURRENCY) -- anything else must signal "can't
    # convert" rather than silently treating it as 1:1 with USD.
    assert get_usd_conversion_rate("EUR") is None


def test_usd_conversion_rate_is_none_when_the_fx_quote_fails(monkeypatch):
    def raise_not_found(ticker):
        raise market_data.TickerNotFoundError(ticker)

    monkeypatch.setattr(market_data, "get_quote", raise_not_found)
    assert get_usd_conversion_rate("INR") is None


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


# ---------------------------------------------------------------- get_corporate_actions_history

@pytest.fixture(autouse=True)
def _clear_corporate_actions_history_cache(monkeypatch):
    monkeypatch.setattr(md, "_corporate_actions_history_cache", {})


def test_get_corporate_actions_history_returns_full_dividend_and_split_lists(monkeypatch):
    dividends = pd.Series([0.22, 0.24, 0.27], index=pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01"]))
    splits = pd.Series([7.0, 4.0], index=pd.to_datetime(["2014-06-09", "2020-08-31"]))
    stock = _FakeCorporateActionsStock(dividends=dividends, splits=splits)
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions_history("AAPL")

    assert result["dividends"] == [
        {"date": "2025-01-01", "amount": 0.22},
        {"date": "2025-04-01", "amount": 0.24},
        {"date": "2025-07-01", "amount": 0.27},
    ]
    assert result["splits"] == [
        {"date": "2014-06-09", "ratio": 7.0},
        {"date": "2020-08-31", "ratio": 4.0},
    ]


def test_get_corporate_actions_history_returns_empty_lists_for_no_history(monkeypatch):
    stock = _FakeCorporateActionsStock()
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: stock)

    result = md.get_corporate_actions_history("NEWCO")

    assert result["dividends"] == []
    assert result["splits"] == []
    assert result["next_earnings_date"] is None


def test_get_corporate_actions_history_is_cached_within_the_ttl(monkeypatch):
    calls = []

    def make_stock(ticker):
        calls.append(ticker)
        return _FakeCorporateActionsStock()

    monkeypatch.setattr(md.yf, "Ticker", make_stock)

    md.get_corporate_actions_history("MSFT")
    md.get_corporate_actions_history("MSFT")
    assert len(calls) == 1


# ---------------------------------------------------------------- get_benchmark_history / SECTOR_ETF_PROXIES

class _FakeBenchmarkStock:
    def __init__(self, history_df):
        self._history_df = history_df

    def history(self, period="5y"):
        return self._history_df


@pytest.fixture(autouse=True)
def _clear_benchmark_history_cache(monkeypatch):
    monkeypatch.setattr(md, "_benchmark_history_cache", {})


def test_get_benchmark_history_returns_the_fetched_dataframe(monkeypatch):
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: _FakeBenchmarkStock(df))

    result = md.get_benchmark_history("^GSPC")
    assert list(result["Close"]) == [100.0, 101.0]


def test_get_benchmark_history_is_cached_within_the_ttl(monkeypatch):
    calls = []

    def make_stock(ticker):
        calls.append(ticker)
        return _FakeBenchmarkStock(pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2024-01-01"])))

    monkeypatch.setattr(md.yf, "Ticker", make_stock)

    md.get_benchmark_history("^GSPC")
    md.get_benchmark_history("^GSPC")
    assert len(calls) == 1


def test_get_benchmark_history_caches_different_tickers_independently(monkeypatch):
    monkeypatch.setattr(
        md.yf, "Ticker",
        lambda ticker: _FakeBenchmarkStock(pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2024-01-01"]))),
    )
    md.get_benchmark_history("^GSPC")
    md.get_benchmark_history("^TNX")
    assert set(md._benchmark_history_cache.keys()) == {"^GSPC", "^TNX"}


def test_get_benchmark_history_returns_none_on_empty_data(monkeypatch):
    monkeypatch.setattr(md.yf, "Ticker", lambda ticker: _FakeBenchmarkStock(pd.DataFrame()))
    assert md.get_benchmark_history("^GSPC") is None


def test_get_benchmark_history_returns_none_instead_of_raising(monkeypatch):
    def raise_error(ticker):
        raise ValueError("network error")

    monkeypatch.setattr(md.yf, "Ticker", raise_error)
    assert md.get_benchmark_history("^GSPC") is None


def test_sector_etf_proxies_cover_every_yfinance_sector_confirmed_live():
    # Confirmed via a live fetch (AAPL/JPM/XOM/JNJ/PG/AMZN/NEE/AMT/CAT/
    # DIS/LIN) before this dict was written -- see market_data.py's own
    # comment. Guards against a future edit silently dropping one of
    # these 11 real yfinance sector strings.
    expected_sectors = {
        "Technology", "Financial Services", "Energy", "Healthcare",
        "Consumer Defensive", "Consumer Cyclical", "Utilities",
        "Real Estate", "Industrials", "Communication Services", "Basic Materials",
    }
    assert set(md.SECTOR_ETF_PROXIES.keys()) == expected_sectors


# ---------------------------------------------------------------- quarterly statements

class _FakeQuarterlyStock:
    def __init__(self, quarterly_financials=None, quarterly_balance_sheet=None, quarterly_cashflow=None):
        self.quarterly_financials = quarterly_financials if quarterly_financials is not None else pd.DataFrame()
        self.quarterly_balance_sheet = quarterly_balance_sheet if quarterly_balance_sheet is not None else pd.DataFrame()
        self.quarterly_cashflow = quarterly_cashflow if quarterly_cashflow is not None else pd.DataFrame()


def _loader_with_quarterly(**kwargs):
    loader = MarketDataLoader("MSFT")
    loader.stock = _FakeQuarterlyStock(**kwargs)
    return loader


def test_get_quarterly_income_statement_returns_the_raw_dataframe():
    df = pd.DataFrame({"2024-09-30": [100]}, index=["Total Revenue"])
    loader = _loader_with_quarterly(quarterly_financials=df)
    result = loader.get_quarterly_income_statement()
    assert list(result.loc["Total Revenue"]) == [100]


def test_get_quarterly_balance_sheet_returns_the_raw_dataframe():
    df = pd.DataFrame({"2024-09-30": [500]}, index=["Stockholders Equity"])
    loader = _loader_with_quarterly(quarterly_balance_sheet=df)
    result = loader.get_quarterly_balance_sheet()
    assert list(result.loc["Stockholders Equity"]) == [500]


def test_get_quarterly_cash_flow_returns_the_raw_dataframe():
    df = pd.DataFrame({"2024-09-30": [20]}, index=["Operating Cash Flow"])
    loader = _loader_with_quarterly(quarterly_cashflow=df)
    result = loader.get_quarterly_cash_flow()
    assert list(result.loc["Operating Cash Flow"]) == [20]


def test_quarterly_statement_methods_raise_on_empty_dataframe():
    # Same "empty DataFrame -> ValueError" contract as the annual
    # statement methods (_cached_statement's shared error path).
    loader = _loader_with_quarterly()
    try:
        loader.get_quarterly_income_statement()
        assert False, "expected ValueError"
    except ValueError:
        pass
