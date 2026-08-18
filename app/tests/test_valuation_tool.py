"""
Unit tests for ValuationTool (app/tools/valuation_tool.py) -- runs the
real, unmocked valuation engines (ValuationPipeline, RelativeValuationEngine,
MonteCarloDCFEngine, the real committed ML classifier) against a
realistic hand-built ResearchContext. The only mocked seam is network:
get_benchmark_history (app/data/market_data.py's yfinance-backed
S&P/rate/sector history fetch), used by AlphaFactorsEngine.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.research_context import ResearchContext
from app.tools import valuation_tool as vt
from app.tools.valuation_tool import ValuationTool


def _financial_df(rows=2):
    base = {
        "revenue": [90, 100, 110][-rows:], "ebit": [18, 20, 22][-rows:], "net_income": [13, 15, 16][-rows:],
        "cash_from_operations": [16, 18, 19][-rows:], "capex": [4, 5, 5][-rows:],
        "total_debt": [45, 50, 55][-rows:], "tax_expense": [2, 3, 3][-rows:], "pretax_income": [16, 18, 19][-rows:],
        "depreciation": [3, 4, 4][-rows:], "current_assets": [27, 30, 32][-rows:], "current_liabilities": [18, 20, 21][-rows:],
        "cash": [38, 40, 42][-rows:], "shares_outstanding": [1000, 1000, 1000][-rows:], "interest_expense": [-2, -2, -2][-rows:],
        "total_equity": [75, 80, 85][-rows:],
    }
    index = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])[-rows:]
    return pd.DataFrame(base, index=index)


def _context(ticker="TEST", historical_prices=None, market_cap=5000, current_price=50.0, rows=2):
    context = ResearchContext(ticker=ticker, question="Is this undervalued?")
    context.normalized_financials = _financial_df(rows=rows)
    context.market_cap = market_cap
    context.beta = 1.1
    context.company_info = {"current_price": current_price, "currency": "USD", "sector": None}
    context.historical_prices = historical_prices
    return context


@pytest.fixture(autouse=True)
def _no_network_benchmark_history(monkeypatch):
    # AlphaFactorsEngine's benchmark/sector/rate comparisons hit
    # get_benchmark_history (real yfinance network call) for any
    # non-.NS ticker -- the true external seam, mocked to keep every
    # test in this file network-free and deterministic.
    monkeypatch.setattr(vt, "get_benchmark_history", lambda ticker: None)


# ---------------------------------------------------------------- happy path

def test_run_populates_valuation_results_and_context_dcf_fields():
    context = _context()
    result = ValuationTool().run(context)

    assert result.valuation_results["dcf_available"] is True
    assert result.valuation_results["dcf_unavailable_reason"] is None
    assert result.enterprise_value == result.valuation_results["enterprise_value"]
    assert result.equity_value == result.valuation_results["equity_value"]
    assert result.intrinsic_value == result.valuation_results["intrinsic_value"]
    assert result.enterprise_value is not None
    assert result.equity_value is not None
    assert result.intrinsic_value is not None
    assert result.intrinsic_value == pytest.approx(result.equity_value / 1000)  # shares_outstanding=1000


def test_run_computes_current_price_and_upside_from_intrinsic_value():
    context = _context(current_price=50.0)
    result = ValuationTool().run(context)

    assert result.valuation_results["current_price"] == 50.0
    expected_upside = (result.intrinsic_value - 50.0) / 50.0 * 100
    assert result.valuation_results["upside_percent"] == pytest.approx(round(expected_upside, 2))


def test_run_records_the_tool_in_the_trace():
    context = _context()
    result = ValuationTool().run(context)
    assert "valuation_tool" in result.tool_trace


def test_run_attaches_a_monte_carlo_distribution_when_shares_outstanding_is_known():
    context = _context()
    result = ValuationTool().run(context)

    mc = result.valuation_results["monte_carlo"]
    assert mc is not None
    assert mc["ci_lower"] <= mc["p25"] <= mc["median"] <= mc["p75"] <= mc["ci_upper"]
    assert 0.0 <= mc["prob_undervalued"] <= 1.0


def test_run_attaches_alpha_factors_scorecard_for_a_us_ticker():
    context = _context(ticker="TEST")
    result = ValuationTool().run(context)
    alpha = result.valuation_results["alpha_factors"]
    assert set(alpha.keys()) == {"financial", "quality", "valuation", "market", "risk", "sentiment", "macro"}


# ---------------------------------------------------------------- relative valuation + ML classifier, wired together

def test_run_populates_relative_valuation_and_ml_classifier_when_price_history_is_available():
    # With three fiscal years of financials AND a matching multi-year
    # price history, RelativeValuationEngine can form a signal, which
    # in turn supplies ml_features.extract_features' relative_vs_history_pct
    # -- the one feature missing in the no-history case below -- so the
    # real committed ML classifier produces an actual verdict.
    dates = pd.date_range("2022-01-01", "2025-01-01", freq="D")
    prices = pd.DataFrame({"Close": np.linspace(40, 55, len(dates))}, index=dates)
    context = _context(rows=3, historical_prices=prices, market_cap=55000, current_price=55.0)

    result = ValuationTool().run(context)

    relative = result.valuation_results["relative_valuation"]
    assert relative is not None
    assert relative["signal"] in ("cheap", "expensive", "in-line")

    ml = result.valuation_results["ml_classifier"]
    assert ml is not None
    assert ml["verdict"] in ("UNDERVALUED", "FAIRLY VALUED", "OVERVALUED")
    assert sum(ml["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_run_leaves_relative_valuation_and_ml_classifier_none_without_price_history():
    # No historical_prices -> RelativeValuationEngine can't compute a
    # year-end price for any fiscal year -> evaluate() returns None ->
    # extract_features' relative_vs_history_pct feature is None ->
    # predict_verdict's missing-feature guard returns None too. A
    # chain of graceful degradation, not a crash.
    context = _context(historical_prices=None)
    result = ValuationTool().run(context)

    assert result.valuation_results["relative_valuation"] is None
    assert result.valuation_results["ml_classifier"] is None
    # DCF itself is unaffected by the missing price history.
    assert result.valuation_results["dcf_available"] is True


# ---------------------------------------------------------------- guard: normalized_financials is None

def test_run_invokes_market_data_tool_when_normalized_financials_is_none(monkeypatch):
    calls = []

    class _FakeMarketDataTool:
        def run(self, context):
            calls.append(context.ticker)
            context.normalized_financials = _financial_df(rows=2)
            context.market_cap = 5000
            context.beta = 1.1
            context.company_info = {"current_price": 50.0, "currency": "USD", "sector": None}
            context.historical_prices = None
            return context

    monkeypatch.setattr("app.tools.market_data_tool.MarketDataTool", _FakeMarketDataTool)

    context = ResearchContext(ticker="TEST", question="Is this undervalued?")
    assert context.normalized_financials is None

    result = ValuationTool().run(context)

    assert calls == ["TEST"]  # market_data_tool.py's fallback ran exactly once
    assert result.valuation_results["dcf_available"] is True


def test_run_skips_market_data_tool_when_normalized_financials_already_populated(monkeypatch):
    calls = []

    class _FakeMarketDataTool:
        def run(self, context):
            calls.append(context.ticker)
            return context

    monkeypatch.setattr("app.tools.market_data_tool.MarketDataTool", _FakeMarketDataTool)

    context = _context()  # normalized_financials already set
    ValuationTool().run(context)

    assert calls == []  # guard at the top of run() must not re-fetch


# ---------------------------------------------------------------- crypto path

def test_run_returns_the_crypto_unavailable_shape_for_a_crypto_ticker():
    context = ResearchContext(ticker="BTC-USD", question="Is BTC undervalued?")
    context.normalized_financials = pd.DataFrame()  # market_data_tool.py's own crypto shape
    context.market_cap = 900_000_000_000
    context.beta = 1.2
    context.company_info = {"current_price": 60000.0, "currency": "USD"}
    context.historical_prices = None

    result = ValuationTool().run(context)

    assert result.valuation_results["dcf_available"] is False
    assert result.valuation_results["is_crypto"] is True
    assert "cryptocurrenc" in result.valuation_results["dcf_unavailable_reason"].lower()
    assert result.valuation_results["current_price"] == 60000.0
    assert result.enterprise_value is None
    assert result.equity_value is None
    assert result.intrinsic_value is None
    assert "valuation_tool" in result.tool_trace


def test_run_crypto_path_does_not_call_market_data_tool_when_already_populated(monkeypatch):
    calls = []

    class _FakeMarketDataTool:
        def run(self, context):
            calls.append(context.ticker)
            return context

    monkeypatch.setattr("app.tools.market_data_tool.MarketDataTool", _FakeMarketDataTool)

    context = ResearchContext(ticker="BTC-USD", question="q")
    context.normalized_financials = pd.DataFrame()
    context.company_info = {"current_price": 60000.0}

    ValuationTool().run(context)
    assert calls == []
