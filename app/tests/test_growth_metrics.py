"""
Unit tests for app/analysis/growth_metrics.py -- builds fake raw
statement DataFrames (yfinance's own shape: rows=line items,
columns=fiscal year-end dates), matching test_financial_normalizer.py's
convention. No network -- MarketDataLoader's statement methods are
monkeypatched directly on the instance.
"""

import pandas as pd
import pytest

from app.analysis import growth_metrics
from app.data.market_data import MarketDataLoader


def _statement(index_labels, columns, data):
    return pd.DataFrame(data, index=index_labels, columns=columns)


def _patch_annual_and_quarterly(monkeypatch, ticker, income, balance, cash_flow):
    def _fake_init(self, t):
        self.ticker = t.upper()

    monkeypatch.setattr(MarketDataLoader, "__init__", _fake_init)
    monkeypatch.setattr(MarketDataLoader, "get_income_statement", lambda self: income)
    monkeypatch.setattr(MarketDataLoader, "get_balance_sheet", lambda self: balance)
    monkeypatch.setattr(MarketDataLoader, "get_cash_flow", lambda self: cash_flow)
    monkeypatch.setattr(MarketDataLoader, "get_quarterly_income_statement", lambda self: income)
    monkeypatch.setattr(MarketDataLoader, "get_quarterly_balance_sheet", lambda self: balance)
    monkeypatch.setattr(MarketDataLoader, "get_quarterly_cash_flow", lambda self: cash_flow)


# ---------------------------------------------------------------- _compute_cagr

def test_compute_cagr_over_exactly_enough_history():
    series = pd.Series([100.0, 110.0, 121.0, 133.1])  # 3 periods of +10%
    assert growth_metrics._compute_cagr(series, 3) == 10.0


def test_compute_cagr_returns_none_with_insufficient_history():
    series = pd.Series([100.0, 110.0])  # only 1 period, need 3
    assert growth_metrics._compute_cagr(series, 3) is None


def test_compute_cagr_returns_none_for_non_positive_start():
    series = pd.Series([-50.0, 10.0, 20.0, 30.0])
    assert growth_metrics._compute_cagr(series, 3) is None


def test_compute_cagr_returns_none_for_non_positive_end():
    series = pd.Series([50.0, 10.0, 20.0, -5.0])
    assert growth_metrics._compute_cagr(series, 3) is None


def test_compute_cagr_ignores_nan_when_dropping():
    series = pd.Series([100.0, None, 110.0, 121.0])
    # after dropna: [100, 110, 121] -- exactly 2 periods
    assert growth_metrics._compute_cagr(series, 2) == pytest.approx(10.0)


# ---------------------------------------------------------------- build_growth_metrics

def test_build_growth_metrics_computes_revenue_eps_book_value_fcf_cagr(monkeypatch):
    years = ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]
    income = _statement(
        ["Total Revenue", "Net Income"],
        years,
        [[100, 110, 121, 133.1], [10, 11, 12.1, 13.31]],
    )
    balance = _statement(
        ["Stockholders Equity", "Ordinary Shares Number"],
        years,
        [[500, 550, 605, 665.5], [10, 10, 10, 10]],
    )
    cash_flow = _statement(
        ["Operating Cash Flow", "Capital Expenditure"],
        years,
        [[20, 22, 24.2, 26.6], [5, 5, 5, 5]],
    )
    _patch_annual_and_quarterly(monkeypatch, "TEST", income, balance, cash_flow)

    result = growth_metrics.build_growth_metrics("TEST")

    assert result["revenue_cagr"]["3y"] == 10.0
    assert result["revenue_cagr"]["1y"] is not None
    assert result["revenue_cagr"]["5y"] is None  # only 4 years of data
    assert result["eps_cagr"]["3y"] is not None
    assert result["book_value_cagr"]["3y"] is not None
    assert result["fcf_cagr"]["1y"] is not None


def test_build_growth_metrics_degrades_gracefully_on_empty_statements(monkeypatch):
    empty = pd.DataFrame()
    _patch_annual_and_quarterly(monkeypatch, "EMPTY", empty, empty, empty)

    result = growth_metrics.build_growth_metrics("EMPTY")

    assert result["revenue_cagr"] == {"1y": None, "3y": None, "5y": None}
    assert result["eps_cagr"] == {"1y": None, "3y": None, "5y": None}


# ---------------------------------------------------------------- build_financial_performance

def test_build_financial_performance_computes_margins(monkeypatch):
    years = ["2023-12-31", "2024-12-31"]
    income = _statement(
        ["Total Revenue", "Net Income", "EBIT"],
        years,
        [[100, 200], [10, 30], [20, 60]],
    )
    _patch_annual_and_quarterly(monkeypatch, "TEST", income, pd.DataFrame(), pd.DataFrame())

    rows = growth_metrics.build_financial_performance("TEST", period="yearly")

    assert len(rows) == 2
    assert rows[0]["period_end"] == "2023-12-31"
    assert rows[0]["revenue"] == 100.0
    assert rows[0]["net_margin_pct"] == 10.0
    assert rows[0]["operating_margin_pct"] == 20.0
    assert rows[1]["net_margin_pct"] == 15.0


def test_build_financial_performance_handles_zero_revenue_without_crashing(monkeypatch):
    income = _statement(["Total Revenue", "Net Income"], ["2024-12-31"], [[0], [5]])
    _patch_annual_and_quarterly(monkeypatch, "TEST", income, pd.DataFrame(), pd.DataFrame())

    rows = growth_metrics.build_financial_performance("TEST", period="yearly")

    assert rows[0]["net_margin_pct"] is None


def test_build_financial_performance_uses_quarterly_statements_when_requested(monkeypatch):
    annual_income = _statement(["Total Revenue"], ["2024-12-31"], [[999]])
    quarterly_income = _statement(["Total Revenue"], ["2024-09-30", "2024-12-31"], [[40, 45]])

    def _fake_init(self, t):
        self.ticker = t.upper()

    monkeypatch.setattr(MarketDataLoader, "__init__", _fake_init)
    monkeypatch.setattr(MarketDataLoader, "get_income_statement", lambda self: annual_income)
    monkeypatch.setattr(MarketDataLoader, "get_balance_sheet", lambda self: pd.DataFrame())
    monkeypatch.setattr(MarketDataLoader, "get_cash_flow", lambda self: pd.DataFrame())
    monkeypatch.setattr(MarketDataLoader, "get_quarterly_income_statement", lambda self: quarterly_income)
    monkeypatch.setattr(MarketDataLoader, "get_quarterly_balance_sheet", lambda self: pd.DataFrame())
    monkeypatch.setattr(MarketDataLoader, "get_quarterly_cash_flow", lambda self: pd.DataFrame())

    rows = growth_metrics.build_financial_performance("TEST", period="quarterly")

    assert len(rows) == 2
    assert rows[-1]["revenue"] == 45.0


def test_build_growth_metrics_degrades_cleanly_for_a_crypto_ticker():
    result = growth_metrics.build_growth_metrics("BTC-USD")
    assert result == {
        "revenue_cagr": {"1y": None, "3y": None, "5y": None},
        "eps_cagr": {"1y": None, "3y": None, "5y": None},
        "book_value_cagr": {"1y": None, "3y": None, "5y": None},
        "fcf_cagr": {"1y": None, "3y": None, "5y": None},
    }


def test_build_financial_performance_degrades_cleanly_for_a_crypto_ticker():
    assert growth_metrics.build_financial_performance("ETH-USD") == []
