"""
Unit tests for app/reasoning/model_consensus.py. HostedProvider itself
is never hit -- model_consensus.HostedProvider is monkeypatched to a
fake class per test, since these tests are about this module's own
prompt-building/parsing/tallying logic, not HostedProvider's HTTP
behavior (already covered by test_llm_provider.py).
"""

import pandas as pd
import pytest

from app.core.llm_provider import LLMProviderError
from app.core.research_context import ResearchContext
from app.reasoning import model_consensus as mc
from app.tools.valuation_tool import ValuationTool


def _report_data(**overrides):
    base = {
        "recommendation": {"rating": "Buy", "basis": "DCF implies 20% upside"},
        "valuation_analysis": {
            "Intrinsic Value (per share)": 150.0,
            "Current Price": 125.0,
            "Upside (%)": 20.0,
        },
        "market_earnings_snapshot": {
            "sentiment_label": "Positive",
            "sentiment_confidence": "high",
            "news_sentiment_label": "Mixed",
            "news_sentiment_confidence": "medium",
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- _parse_opinion

def test_parse_opinion_extracts_rating_confidence_and_reasoning():
    raw = "RATING: Buy\nCONFIDENCE: 78\nREASONING: Upside is well above the WACC hurdle."
    result = mc._parse_opinion(raw)
    assert result == {"rating": "Buy", "confidence": 78, "reasoning": "Upside is well above the WACC hurdle."}


def test_parse_opinion_is_case_insensitive_on_rating():
    raw = "RATING: sell\nCONFIDENCE: 60\nREASONING: Overvalued."
    assert mc._parse_opinion(raw)["rating"] == "Sell"


def test_parse_opinion_clamps_out_of_range_confidence():
    raw = "RATING: Hold\nCONFIDENCE: 140\nREASONING: Mixed signals."
    assert mc._parse_opinion(raw)["confidence"] == 100


def test_parse_opinion_degrades_to_insufficient_data_on_garbage():
    result = mc._parse_opinion("I refuse to give financial advice.")
    assert result == {"rating": "Insufficient Data", "confidence": None, "reasoning": "Model response could not be parsed."}


def test_parse_opinion_missing_confidence_is_none_not_a_crash():
    raw = "RATING: Buy\nREASONING: Strong fundamentals."
    result = mc._parse_opinion(raw)
    assert result["rating"] == "Buy"
    assert result["confidence"] is None


def test_parse_opinion_truncates_a_runaway_reasoning_line():
    raw = "RATING: Buy\nCONFIDENCE: 80\nREASONING: " + ("x" * 500)
    assert len(mc._parse_opinion(raw)["reasoning"]) == 300


# ---------------------------------------------------------------- _build_prompt

def test_build_prompt_includes_ticker_and_key_figures():
    prompt = mc._build_prompt(_report_data(), "AAPL")
    assert "TICKER: AAPL" in prompt
    assert "150.0" in prompt
    assert "Positive" in prompt


def test_build_prompt_degrades_missing_fields_to_unavailable():
    sparse = {"recommendation": {}, "valuation_analysis": {}, "market_earnings_snapshot": {}}
    prompt = mc._build_prompt(sparse, "ZZZZ")
    assert "Unavailable" in prompt


# ---------------------------------------------------------------- get_model_opinions

class _FakeProvider:
    def __init__(self, model=None, **kwargs):
        self.model = model

    def generate(self, prompt, max_new_tokens=150):
        return f"RATING: Buy\nCONFIDENCE: 70\nREASONING: Fake opinion from {self.model}."


class _FlakyProvider:
    """One specific model (by name) raises; the others behave normally --
    exercises that one model's outage doesn't take down the other two."""
    def __init__(self, model=None, **kwargs):
        self.model = model

    def generate(self, prompt, max_new_tokens=150):
        if self.model == "openai/gpt-oss-120b":
            raise LLMProviderError("simulated outage")
        return "RATING: Hold\nCONFIDENCE: 55\nREASONING: Neutral."


def test_get_model_opinions_returns_one_entry_per_configured_model(monkeypatch):
    monkeypatch.setattr(mc, "HostedProvider", _FakeProvider)
    opinions = mc.get_model_opinions(_report_data(), "AAPL")
    assert len(opinions) == len(mc.CONSENSUS_MODELS)
    assert {o["model"] for o in opinions} == {m["model"] for m in mc.CONSENSUS_MODELS}
    assert all(o["rating"] == "Buy" for o in opinions)


def test_get_model_opinions_isolates_one_models_outage(monkeypatch):
    monkeypatch.setattr(mc, "HostedProvider", _FlakyProvider)
    opinions = mc.get_model_opinions(_report_data(), "AAPL")
    by_model = {o["model"]: o for o in opinions}
    assert by_model["openai/gpt-oss-120b"]["rating"] == "Insufficient Data"
    assert by_model["llama-3.3-70b-versatile"]["rating"] == "Hold"
    assert by_model["qwen/qwen3.6-27b"]["rating"] == "Hold"


# ---------------------------------------------------------------- compute_consensus

def test_compute_consensus_majority_rules():
    opinions = [
        {"rating": "Buy"}, {"rating": "Buy"}, {"rating": "Hold"},
    ]
    assert mc.compute_consensus(opinions) == {"rating": "Buy", "agree_count": 2, "total": 3}


def test_compute_consensus_unanimous():
    opinions = [{"rating": "Sell"}, {"rating": "Sell"}, {"rating": "Sell"}]
    assert mc.compute_consensus(opinions) == {"rating": "Sell", "agree_count": 3, "total": 3}


def test_compute_consensus_all_insufficient_data_degrades_cleanly():
    opinions = [{"rating": "Insufficient Data"}] * 3
    assert mc.compute_consensus(opinions) == {"rating": "Insufficient Data", "agree_count": 0, "total": 3}


def test_compute_consensus_ignores_insufficient_data_when_tallying():
    opinions = [{"rating": "Buy"}, {"rating": "Insufficient Data"}, {"rating": "Buy"}]
    result = mc.compute_consensus(opinions)
    assert result["rating"] == "Buy"
    assert result["agree_count"] == 2
    assert result["total"] == 3


# ---------------------------------------------------------------- get_stock_model_opinions

def _valid_financial_df():
    years = pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"])
    return pd.DataFrame({
        "revenue": [100, 115, 132, 152], "ebit": [20, 24, 28, 33], "net_income": [15, 18, 21, 25],
        "cash_from_operations": [18, 21, 24, 28], "capex": [5, 5, 6, 6], "total_debt": [50, 52, 53, 54],
        "tax_expense": [3, 3, 4, 4], "pretax_income": [18, 21, 25, 29], "depreciation": [4, 4, 5, 5],
        "current_assets": [30, 34, 38, 43], "current_liabilities": [20, 21, 22, 23], "cash": [40, 44, 48, 53],
        "shares_outstanding": [1000, 1000, 1000, 1000], "interest_expense": [-2, -2, -2, -2],
        "total_equity": [80, 92, 106, 122], "total_assets": [200, 220, 245, 275], "retained_earnings": [60, 70, 82, 96],
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


def test_get_stock_model_opinions_returns_models_and_consensus(monkeypatch):
    ctx = _valid_context()
    monkeypatch.setattr(mc, "ResearchContext", lambda ticker, question: ctx)
    monkeypatch.setattr(mc, "HostedProvider", _FakeProvider)

    result = mc.get_stock_model_opinions("TEST.NS")

    assert len(result["models"]) == len(mc.CONSENSUS_MODELS)
    assert result["consensus"]["rating"] == "Buy"


def test_get_stock_model_opinions_propagates_ticker_not_found(monkeypatch):
    from app.data.market_data import TickerNotFoundError

    def _raise_run(self, context):
        raise TickerNotFoundError("bad ticker")

    monkeypatch.setattr(ValuationTool, "run", _raise_run)

    with pytest.raises(TickerNotFoundError):
        mc.get_stock_model_opinions("ZZZZ")
