"""
Unit tests for app/reasoning/model_consensus.py. HostedProvider itself
is never hit -- model_consensus.HostedProvider is monkeypatched to a
fake class per test, since these tests are about this module's own
prompt-building/parsing/tallying logic, not HostedProvider's HTTP
behavior (already covered by test_llm_provider.py).
"""

import pytest

from app.core.llm_provider import LLMProviderError
from app.reasoning import model_consensus as mc


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
