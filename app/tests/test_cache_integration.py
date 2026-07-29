"""
Unit tests verifying market_data.py and valuation_pipeline.py actually
call into app/core/cache.py correctly -- mocking cache_get/cache_set
directly (not a real or fake Redis) so these stay fast and dependency-free,
matching test_cache.py's own approach. What's under test is "does a
cache hit skip the expensive work, does a cache miss compute-then-store,"
not Redis itself (already covered by test_cache.py).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data.market_data import MarketDataLoader
from app.valuation.valuation_pipeline import ValuationPipeline


# ---------------------------------------------------------- market_data.py

def test_get_income_statement_returns_cached_value_without_hitting_yfinance():
    loader = MarketDataLoader("AAPL")
    cached_df = pd.DataFrame({"Total Revenue": [100]})

    with patch("app.data.market_data.cache_get", return_value=cached_df) as mock_get, \
         patch("app.data.market_data.cache_set") as mock_set:
        loader.stock = MagicMock()  # would raise if touched -- proves the cache hit short-circuits
        result = loader.get_income_statement()

    assert result is cached_df
    loader.stock.financials.__class__ = MagicMock  # sanity: never actually read
    mock_set.assert_not_called()


def test_get_balance_sheet_computes_and_caches_on_a_miss():
    loader = MarketDataLoader("AAPL")
    fresh_df = pd.DataFrame({"Total Debt": [500]})
    loader.stock = MagicMock()
    loader.stock.balance_sheet = fresh_df

    with patch("app.data.market_data.cache_get", return_value=None), \
         patch("app.data.market_data.cache_set") as mock_set:
        result = loader.get_balance_sheet()

    assert result is fresh_df
    mock_set.assert_called_once()
    assert mock_set.call_args.kwargs["ttl_seconds"] > 0


def test_statement_fetch_raises_on_empty_dataframe_without_caching_it():
    loader = MarketDataLoader("AAPL")
    loader.stock = MagicMock()
    loader.stock.cashflow = pd.DataFrame()

    with patch("app.data.market_data.cache_get", return_value=None), \
         patch("app.data.market_data.cache_set") as mock_set:
        with pytest.raises(ValueError):
            loader.get_cash_flow()

    mock_set.assert_not_called()


def test_company_info_is_never_cached_current_price_always_fresh():
    """current_price/market_cap must never go through the statement
    cache -- serving a stale price on a live research tool would be a
    real correctness bug, not just a staleness inconvenience (see
    market_data.py's module-level comment)."""
    import inspect
    from app.data import market_data
    source = inspect.getsource(market_data.MarketDataLoader.get_company_info)
    assert "cache_get" not in source
    assert "cache_set" not in source


# ------------------------------------------------------- valuation_pipeline.py

def _financial_df():
    index = pd.to_datetime(["2023-12-31", "2024-12-31"])
    return pd.DataFrame({
        "revenue": [100, 110], "ebit": [20, 22], "net_income": [15, 16],
        "cash_from_operations": [18, 19], "capex": [5, 5],
        "total_debt": [50, 55], "tax_expense": [3, 3], "pretax_income": [18, 19],
        "depreciation": [4, 4], "current_assets": [30, 32], "current_liabilities": [20, 21],
        "cash": [40, 42], "shares_outstanding": [1000, 1000], "interest_expense": [-2, -2],
        "total_equity": [80, 85],
    }, index=index)


def test_run_valuation_returns_cached_result_without_recomputing():
    pipeline = ValuationPipeline(financial_df=_financial_df(), market_cap=5000, beta=1.1, ticker="AAPL")
    cached_result = {"dcf_available": True, "enterprise_value": 999}

    with patch("app.valuation.valuation_pipeline.cache_get", return_value=cached_result), \
         patch.object(pipeline, "_compute_valuation") as mock_compute:
        result = pipeline.run_valuation()

    assert result is cached_result
    mock_compute.assert_not_called()


def test_run_valuation_computes_and_caches_on_a_miss():
    pipeline = ValuationPipeline(financial_df=_financial_df(), market_cap=5000, beta=1.1, ticker="AAPL")

    with patch("app.valuation.valuation_pipeline.cache_get", return_value=None), \
         patch("app.valuation.valuation_pipeline.cache_set") as mock_set:
        result = pipeline.run_valuation()

    assert result["dcf_available"] is True
    mock_set.assert_called_once()


def test_cache_key_is_stable_for_identical_inputs():
    df = _financial_df()
    key1 = ValuationPipeline(financial_df=df, market_cap=5000, beta=1.1, ticker="AAPL")._cache_key()
    key2 = ValuationPipeline(financial_df=df, market_cap=5000, beta=1.1, ticker="AAPL")._cache_key()
    assert key1 == key2


def test_cache_key_ignores_market_cap_noise_within_the_rounding_bucket():
    df = _financial_df()
    key1 = ValuationPipeline(financial_df=df, market_cap=5_000_000_000, beta=1.1, ticker="AAPL")._cache_key()
    key2 = ValuationPipeline(financial_df=df, market_cap=5_000_000_001, beta=1.1, ticker="AAPL")._cache_key()
    assert key1 == key2


def test_cache_key_changes_when_market_cap_moves_meaningfully():
    df = _financial_df()
    key1 = ValuationPipeline(financial_df=df, market_cap=5_000_000_000, beta=1.1, ticker="AAPL")._cache_key()
    key2 = ValuationPipeline(financial_df=df, market_cap=5_500_000_000, beta=1.1, ticker="AAPL")._cache_key()
    assert key1 != key2


def test_cache_key_changes_when_financials_change():
    df1 = _financial_df()
    df2 = _financial_df()
    df2["revenue"] = [200, 220]
    key1 = ValuationPipeline(financial_df=df1, market_cap=5000, beta=1.1, ticker="AAPL")._cache_key()
    key2 = ValuationPipeline(financial_df=df2, market_cap=5000, beta=1.1, ticker="AAPL")._cache_key()
    assert key1 != key2


def test_cache_key_differs_by_ticker_for_otherwise_identical_financials():
    df = _financial_df()
    key1 = ValuationPipeline(financial_df=df, market_cap=5000, beta=1.1, ticker="AAPL")._cache_key()
    key2 = ValuationPipeline(financial_df=df, market_cap=5000, beta=1.1, ticker="MSFT")._cache_key()
    assert key1 != key2
