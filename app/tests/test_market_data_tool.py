"""
Unit tests for MarketDataTool (app/tools/market_data_tool.py). The only
mocked seam is MarketDataLoader (the true network dependency) --
FinancialStatementNormaliser and FinancialAnalysisBuilder run for
real, unmocked, against fake-but-realistic yfinance-shaped statement
DataFrames.

Priority coverage: the info_fetch_failed distinction added this
session -- a throttled/failed yfinance .info fetch must raise
MarketDataUnavailableError, while a fetch that succeeded but genuinely
found nothing for the ticker must raise TickerNotFoundError. Conflating
the two would tell a user their perfectly valid ticker "doesn't exist"
when the real cause was a transient provider hiccup.
"""

import pandas as pd
import pytest

from app.core.research_context import ResearchContext
from app.data.market_data import MarketDataUnavailableError, TickerNotFoundError
from app.tools import market_data_tool as mdt
from app.tools.market_data_tool import MarketDataTool


class _FakeLoader:
    """Stands in for MarketDataLoader -- the true external (yfinance)
    seam. Every field is independently overridable so each test can
    isolate exactly the condition it's checking."""

    def __init__(
        self,
        ticker,
        company_info=None,
        info_fetch_failed=False,
        income_statement=None,
        balance_sheet=None,
        cash_flow=None,
        historical_prices=None,
        historical_prices_raises=False,
    ):
        self.ticker = ticker
        self._company_info = company_info if company_info is not None else {}
        self.info_fetch_failed = info_fetch_failed
        self._income_statement = income_statement
        self._balance_sheet = balance_sheet
        self._cash_flow = cash_flow
        self._historical_prices = historical_prices
        self._historical_prices_raises = historical_prices_raises

    def get_company_info(self):
        return self._company_info

    def get_income_statement(self):
        return self._income_statement

    def get_balance_sheet(self):
        return self._balance_sheet

    def get_cash_flow(self):
        return self._cash_flow

    def get_historical_prices(self):
        if self._historical_prices_raises:
            raise RuntimeError("simulated historical-price fetch failure")
        return self._historical_prices


def _make_loader_factory(**kwargs):
    """Returns a callable with MarketDataLoader's (ticker) -> instance
    signature, so it can directly replace the class."""
    def factory(ticker):
        return _FakeLoader(ticker, **kwargs)
    return factory


def _valid_statements():
    income = pd.DataFrame(
        {"2023-12-31": [100, 20, 15, 3, 18, -2], "2024-12-31": [110, 22, 16, 3, 19, -2]},
        index=["Total Revenue", "EBIT", "Net Income", "Tax Provision", "Pretax Income", "Interest Expense"],
    )
    balance = pd.DataFrame(
        {"2023-12-31": [50, 40, 30, 20, 1000, 80], "2024-12-31": [55, 42, 32, 21, 1000, 85]},
        index=["Total Debt", "Cash And Cash Equivalents", "Current Assets", "Current Liabilities",
               "Ordinary Shares Number", "Stockholders Equity"],
    )
    cash_flow = pd.DataFrame(
        {"2023-12-31": [18, -5, 4], "2024-12-31": [19, -5, 4]},
        index=["Operating Cash Flow", "Capital Expenditure", "Depreciation"],
    )
    return income, balance, cash_flow


# ---------------------------------------------------------------- info_fetch_failed distinction (priority)

def test_raises_market_data_unavailable_when_info_fetch_itself_failed(monkeypatch):
    # loader.info_fetch_failed=True (set by get_company_info's own
    # retry-wrapped fetch failing) with an empty info dict -- a
    # throttled/transient provider issue, NOT a bad ticker.
    monkeypatch.setattr(mdt, "MarketDataLoader", _make_loader_factory(
        company_info={}, info_fetch_failed=True,
    ))
    context = ResearchContext(ticker="AAPL", question="q")

    with pytest.raises(MarketDataUnavailableError):
        MarketDataTool().run(context)


def test_raises_ticker_not_found_when_info_is_empty_but_fetch_did_not_fail(monkeypatch):
    # Same empty-ish info dict, but info_fetch_failed=False -- yfinance
    # responded fine, there's just nothing for this ticker (invalid,
    # delisted, or mistyped). Must NOT be reported as a provider outage.
    monkeypatch.setattr(mdt, "MarketDataLoader", _make_loader_factory(
        company_info={}, info_fetch_failed=False,
    ))
    context = ResearchContext(ticker="NOTATICKER", question="q")

    with pytest.raises(TickerNotFoundError):
        MarketDataTool().run(context)


def test_raises_ticker_not_found_not_market_data_unavailable_for_a_bad_ticker(monkeypatch):
    # Explicit negative check: confirms the two exceptions are never
    # conflated in either direction.
    monkeypatch.setattr(mdt, "MarketDataLoader", _make_loader_factory(
        company_info={}, info_fetch_failed=False,
    ))
    context = ResearchContext(ticker="NOTATICKER", question="q")

    try:
        MarketDataTool().run(context)
        assert False, "expected TickerNotFoundError"
    except MarketDataUnavailableError:
        assert False, "a genuinely bad ticker must raise TickerNotFoundError, not MarketDataUnavailableError"
    except TickerNotFoundError:
        pass


def test_does_not_raise_when_info_fetch_failed_but_useful_data_is_present(monkeypatch):
    # info_fetch_failed is only consulted when BOTH company_name and
    # current_price are missing -- if a stale/partial info dict still
    # carries a name or price, that's not treated as a failure.
    monkeypatch.setattr(mdt, "MarketDataLoader", _make_loader_factory(
        company_info={"company_name": "Apple Inc.", "current_price": 190.0},
        info_fetch_failed=True,
        income_statement=pd.DataFrame(), balance_sheet=pd.DataFrame(), cash_flow=pd.DataFrame(),
    ))
    context = ResearchContext(ticker="AAPL", question="q")
    # Should not raise at all -- proceeds past the guard.
    result = MarketDataTool().run(context)
    assert result.company_info["company_name"] == "Apple Inc."


# ---------------------------------------------------------------- happy path

def test_run_populates_normalized_financials_market_cap_and_beta(monkeypatch):
    income, balance, cash_flow = _valid_statements()
    monkeypatch.setattr(mdt, "MarketDataLoader", _make_loader_factory(
        company_info={"company_name": "Test Co", "current_price": 50.0, "beta": 1.3, "market_cap": 5000, "currency": "USD"},
        income_statement=income, balance_sheet=balance, cash_flow=cash_flow,
        historical_prices=pd.DataFrame({"Close": [10.0, 11.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])),
    ))
    context = ResearchContext(ticker="TEST", question="q")

    result = MarketDataTool().run(context)

    assert result.market_cap == 5000
    assert result.beta == 1.3
    assert result.normalized_financials is not None
    assert list(result.normalized_financials["revenue"]) == [100, 110]
    assert "market_data_tool" in result.tool_trace


def test_run_defaults_beta_to_1_2_when_yfinance_omits_it(monkeypatch):
    income, balance, cash_flow = _valid_statements()
    monkeypatch.setattr(mdt, "MarketDataLoader", _make_loader_factory(
        company_info={"company_name": "Test Co", "current_price": 50.0, "beta": None, "market_cap": 5000},
        income_statement=income, balance_sheet=balance, cash_flow=cash_flow,
        historical_prices=pd.DataFrame({"Close": [10.0]}, index=pd.to_datetime(["2024-01-01"])),
    ))
    context = ResearchContext(ticker="TEST", question="q")
    result = MarketDataTool().run(context)
    assert result.beta == 1.2


def test_run_never_lets_a_historical_price_failure_break_the_rest_of_the_run(monkeypatch):
    income, balance, cash_flow = _valid_statements()
    monkeypatch.setattr(mdt, "MarketDataLoader", _make_loader_factory(
        company_info={"company_name": "Test Co", "current_price": 50.0, "beta": 1.1, "market_cap": 5000},
        income_statement=income, balance_sheet=balance, cash_flow=cash_flow,
        historical_prices_raises=True,
    ))
    context = ResearchContext(ticker="TEST", question="q")

    result = MarketDataTool().run(context)

    assert result.historical_prices is None
    assert result.normalized_financials is not None  # rest of the run completed


# ---------------------------------------------------------------- crypto path

def test_run_skips_statement_fetches_for_a_crypto_ticker(monkeypatch):
    calls = {"income": 0, "balance": 0, "cash_flow": 0}

    class _CryptoTrackingLoader(_FakeLoader):
        def get_income_statement(self):
            calls["income"] += 1
            return super().get_income_statement()

        def get_balance_sheet(self):
            calls["balance"] += 1
            return super().get_balance_sheet()

        def get_cash_flow(self):
            calls["cash_flow"] += 1
            return super().get_cash_flow()

    monkeypatch.setattr(mdt, "MarketDataLoader", lambda ticker: _CryptoTrackingLoader(
        ticker,
        company_info={"company_name": "Bitcoin", "current_price": 60000.0},
        historical_prices=pd.DataFrame({"Close": [60000.0]}, index=pd.to_datetime(["2024-01-01"])),
    ))
    context = ResearchContext(ticker="BTC-USD", question="q")

    result = MarketDataTool().run(context)

    assert calls == {"income": 0, "balance": 0, "cash_flow": 0}
    assert result.income_statement is None
    assert result.balance_sheet is None
    assert result.cash_flow is None
    # Empty DataFrame (not None) -- signals "ran, nothing to show" so
    # ValuationTool's `normalized_financials is None` guard doesn't
    # trigger a redundant re-run.
    assert result.normalized_financials is not None
    assert result.normalized_financials.empty
    assert result.financial_summary == {}


def test_run_crypto_path_survives_a_historical_price_failure(monkeypatch):
    class _FailingHistoryLoader(_FakeLoader):
        def get_historical_prices(self):
            raise RuntimeError("simulated failure")

    monkeypatch.setattr(mdt, "MarketDataLoader", lambda ticker: _FailingHistoryLoader(
        ticker, company_info={"company_name": "Bitcoin", "current_price": 60000.0},
    ))
    context = ResearchContext(ticker="BTC-USD", question="q")

    result = MarketDataTool().run(context)
    assert result.historical_prices is None
    assert result.normalized_financials.empty
