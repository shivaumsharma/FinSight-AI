"""
Unit tests for app/reasoning/peer_comparison.py -- monkeypatches
get_stock_overview/build_growth_metrics directly (both independently
tested already in test_stock_overview.py/test_growth_metrics.py), no
network.
"""

from app.reasoning import peer_comparison as pc


def _fake_overview(ticker, market_cap, trailing_pe, currency="USD"):
    return {
        "ticker": ticker, "company_name": f"{ticker} Inc.", "currency": currency,
        "market_cap": market_cap, "fundamentals": {"trailing_pe": trailing_pe},
    }


def _fake_growth(revenue_cagr_3y, eps_cagr_3y):
    return {
        "revenue_cagr": {"1y": None, "3y": revenue_cagr_3y, "5y": None},
        "eps_cagr": {"1y": None, "3y": eps_cagr_3y, "5y": None},
        "book_value_cagr": {"1y": None, "3y": None, "5y": None},
        "fcf_cagr": {"1y": None, "3y": None, "5y": None},
    }


def test_build_peer_comparison_returns_primary_and_peer_rows(monkeypatch):
    overviews = {"AAPL": _fake_overview("AAPL", 3_000_000_000_000, 30.0), "MSFT": _fake_overview("MSFT", 2_800_000_000_000, 35.0)}
    growths = {"AAPL": _fake_growth(6.0, 10.0), "MSFT": _fake_growth(12.0, 15.0)}

    monkeypatch.setattr(pc, "get_stock_overview", lambda ticker: overviews[ticker])
    monkeypatch.setattr(pc, "build_growth_metrics", lambda ticker: growths[ticker])

    result = pc.build_peer_comparison("AAPL", "MSFT")

    assert result["primary"]["ticker"] == "AAPL"
    assert result["primary"]["market_cap"] == 3_000_000_000_000
    assert result["primary"]["trailing_pe"] == 30.0
    assert result["primary"]["revenue_cagr_3y"] == 6.0
    assert result["primary"]["eps_cagr_3y"] == 10.0
    assert result["peer"]["ticker"] == "MSFT"
    assert result["peer"]["revenue_cagr_3y"] == 12.0


def test_build_peer_comparison_propagates_ticker_not_found(monkeypatch):
    from app.data.market_data import TickerNotFoundError

    def _raise(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(pc, "get_stock_overview", _raise)

    try:
        pc.build_peer_comparison("ZZZZ", "MSFT")
        assert False, "expected TickerNotFoundError"
    except TickerNotFoundError:
        pass
