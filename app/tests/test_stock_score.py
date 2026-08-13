"""
Unit tests for app/reasoning/stock_score.py.

Pure scoring helpers are tested directly (no mocking needed). The
orchestration test (build_stock_insights) reuses scripts/phase2_backtest.py's
own established pattern: pre-populate a bare ResearchContext's fields
and call ValuationTool().run(ctx) directly, entirely bypassing
MarketDataLoader/yfinance -- a ".NS" ticker is used specifically so
ValuationTool's own is_non_us_listing gate skips every
get_benchmark_history() call (see valuation_tool.py), keeping this
fully network-free without needing to monkeypatch yfinance at all.
"""

import pandas as pd

from app.core.research_context import ResearchContext
from app.reasoning import stock_score as ss
from app.reporting import institutional_ratings as ir_module
from app.tools.valuation_tool import ValuationTool


# ---------------------------------------------------------------- pure helpers

def test_threshold_star_picks_the_first_matching_floor():
    breakpoints = [(20.0, 5), (10.0, 4), (5.0, 3), (0.0, 2)]
    assert ss._threshold_star(25.0, breakpoints) == 5
    assert ss._threshold_star(12.0, breakpoints) == 4
    assert ss._threshold_star(5.0, breakpoints) == 3
    assert ss._threshold_star(0.0, breakpoints) == 2
    assert ss._threshold_star(-5.0, breakpoints) == 1


def test_threshold_star_returns_none_for_none_input():
    assert ss._threshold_star(None, [(20.0, 5)]) is None


def test_average_stars_skips_none_and_rounds():
    assert ss._average_stars([5, 3, None]) == 4  # (5+3)/2 = 4.0
    assert ss._average_stars([None, None]) is None
    assert ss._average_stars([5, 4, 4]) == ss._clip_star((5 + 4 + 4) / 3)


def test_valuation_star_rescales_composite_score_linearly():
    assert ss._valuation_star({"composite_score": 100}) == 5
    assert ss._valuation_star({"composite_score": -100}) == 1
    assert ss._valuation_star({"composite_score": 0}) == 3
    assert ss._valuation_star({"composite_score": None}) is None


def test_financial_health_star_rewards_low_leverage_and_high_coverage():
    strong = ss._financial_health_star({"Debt to Equity": 0.3, "Current Ratio": 2.5, "Interest Coverage Ratio": 10})
    weak = ss._financial_health_star({"Debt to Equity": 4.0, "Current Ratio": 0.3, "Interest Coverage Ratio": 0.5})
    assert strong == 5
    assert weak == 1


def test_financial_health_star_is_none_when_no_inputs_available():
    assert ss._financial_health_star({}) is None


def test_growth_star_thresholds():
    assert ss._growth_star({"Revenue CAGR (%)": 20.0}) == 5
    assert ss._growth_star({"Revenue CAGR (%)": -5.0}) == 1
    assert ss._growth_star({}) is None


def test_risk_star_uses_altman_zone_and_volatility():
    safe = ss._risk_star({"Altman Z-Score Zone": "Safe Zone", "Annualized Volatility (%)": 15.0})
    distressed = ss._risk_star({"Altman Z-Score Zone": "Distress Zone", "Annualized Volatility (%)": 80.0})
    assert safe == 5
    assert distressed == 1


def test_overall_score_averages_available_categories_onto_0_100():
    assert ss._overall_score({"a": 5, "b": 5, "c": None}) == 100
    assert ss._overall_score({"a": 1, "b": 1}) == 20
    assert ss._overall_score({"a": None}) is None


def test_build_flags_reflects_thresholds_not_llm_judgment():
    scores = {"financial_health": 5, "valuation": 5, "momentum": 5, "risk": 5, "growth": 5}
    flags = ss._build_flags(scores, "Buy", None)
    assert "Strong Financials" in flags
    assert "Attractively Valued" in flags
    assert "Positive Momentum" in flags
    assert "Strong Growth" in flags
    assert "Elevated Risk" not in flags


def test_build_flags_includes_high_institutional_agreement(monkeypatch):
    scores = {"financial_health": None, "valuation": None, "momentum": None, "risk": None, "growth": None}
    consensus = {"score": 80, "low_sample_size": False}
    flags = ss._build_flags(scores, "Buy", consensus)
    assert flags == ["High Institutional Agreement"]


# ---------------------------------------------------------------- build_stock_insights (orchestration)

def _valid_financial_df():
    years = pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"])
    return pd.DataFrame({
        "revenue": [100, 115, 132, 152],
        "ebit": [20, 24, 28, 33],
        "net_income": [15, 18, 21, 25],
        "cash_from_operations": [18, 21, 24, 28],
        "capex": [5, 5, 6, 6],
        "total_debt": [50, 52, 53, 54],
        "tax_expense": [3, 3, 4, 4],
        "pretax_income": [18, 21, 25, 29],
        "depreciation": [4, 4, 5, 5],
        "current_assets": [30, 34, 38, 43],
        "current_liabilities": [20, 21, 22, 23],
        "cash": [40, 44, 48, 53],
        "shares_outstanding": [1000, 1000, 1000, 1000],
        "interest_expense": [-2, -2, -2, -2],
        "total_equity": [80, 92, 106, 122],
        "total_assets": [200, 220, 245, 275],
        "retained_earnings": [60, 70, 82, 96],
    }, index=years)


def _valid_historical_prices(n=260, start=100.0):
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = pd.Series([start + i * 0.2 for i in range(n)], index=index)
    return pd.DataFrame({
        "Open": closes, "High": closes + 1, "Low": closes - 1, "Close": closes,
        "Volume": pd.Series([1_000_000] * n, index=index),
    })


def _valid_context(ticker="TEST.NS"):
    ctx = ResearchContext(ticker=ticker, question=f"Should I invest in {ticker}?")
    ctx.normalized_financials = _valid_financial_df()
    ctx.market_cap = 200_000
    ctx.beta = 1.1
    ctx.historical_prices = _valid_historical_prices()
    ctx.company_info = {"current_price": 200.0, "market_cap": 200_000, "beta": 1.1, "currency": "INR", "sector": "Technology"}
    return ctx


def test_build_stock_insights_returns_full_shape(monkeypatch):
    ctx = _valid_context()
    monkeypatch.setattr(ss, "ResearchContext", lambda ticker, question: ctx)
    monkeypatch.setattr(ir_module, "fetch_institutional_ratings", lambda ticker: [])

    result = ss.build_stock_insights("TEST.NS")

    assert result["rating"] in ("Buy", "Hold", "Sell", "Insufficient Data")
    assert set(result["category_scores"].keys()) == {"valuation", "financial_health", "growth", "profitability", "momentum", "risk"}
    assert isinstance(result["flags"], list)
    assert result["consensus"] is None  # no institutional ratings supplied


def test_build_stock_insights_includes_consensus_when_ratings_exist(monkeypatch):
    ctx = _valid_context()
    monkeypatch.setattr(ss, "ResearchContext", lambda ticker, question: ctx)
    fake_ratings = [
        {"firm": "Firm A", "rating": "BUY", "raw_grade": "Buy", "date": "2024-01-01"},
        {"firm": "Firm B", "rating": "BUY", "raw_grade": "Buy", "date": "2024-01-02"},
        {"firm": "Firm C", "rating": "BUY", "raw_grade": "Buy", "date": "2024-01-03"},
    ]
    monkeypatch.setattr(ss, "fetch_institutional_ratings", lambda ticker: fake_ratings)

    result = ss.build_stock_insights("TEST.NS")

    if result["rating"] in ("Buy", "Hold", "Sell"):
        assert result["consensus"] is not None
        assert result["consensus"]["total_count"] == 3


def test_build_stock_insights_propagates_ticker_not_found(monkeypatch):
    def _raise_run(self, context):
        from app.data.market_data import TickerNotFoundError
        raise TickerNotFoundError("bad ticker")

    monkeypatch.setattr(ValuationTool, "run", _raise_run)

    try:
        ss.build_stock_insights("ZZZZ")
        assert False, "expected TickerNotFoundError"
    except Exception as e:
        from app.data.market_data import TickerNotFoundError
        assert isinstance(e, TickerNotFoundError)


# ---------------------------------------------------------------- crypto degrades cleanly, no crash

def test_build_stock_insights_degrades_cleanly_for_a_crypto_ticker(monkeypatch):
    """Crypto has no financial statements -- ValuationTool's own crypto
    guard (see valuation_tool.py) returns a DCF-unavailable-shaped
    result instead of letting MarketDataTool's statement fetch raise.
    This context mirrors exactly what MarketDataTool.run() sets for a
    crypto ticker: empty (not None) normalized_financials, company_info
    with only a current_price -- no financial fields at all."""
    import pandas as pd

    ctx = ResearchContext(ticker="BTC-USD", question="Should I invest in BTC-USD?")
    ctx.normalized_financials = pd.DataFrame()
    ctx.company_info = {"current_price": 63749.01, "currency": "USD"}
    monkeypatch.setattr(ss, "ResearchContext", lambda ticker, question: ctx)
    monkeypatch.setattr(ir_module, "fetch_institutional_ratings", lambda ticker: [])

    result = ss.build_stock_insights("BTC-USD")

    assert result["rating"] == "Insufficient Data"
    assert result["current_price"] == 63749.01
    assert result["fair_value_estimate"] is None
    assert result["overall_score"] is None
    assert all(v is None for v in result["category_scores"].values())
