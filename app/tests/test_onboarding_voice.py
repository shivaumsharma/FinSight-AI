"""
Unit tests for app/reasoning/onboarding_voice.py. HostedProvider is
monkeypatched to a fake class per test, same convention as
test_chat_router.py -- never hits real network.
"""

import pytest

from app.core.llm_provider import LLMProviderError
from app.reasoning import onboarding_voice as ov


class _RaisingProvider:
    def __init__(self, model=None, **kwargs):
        pass

    def generate(self, prompt, max_new_tokens=150):
        raise LLMProviderError("simulated outage")


def _provider_returning(answer_text):
    class _Provider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return f"ANSWER: {answer_text}"

    return _Provider


def test_classify_onboarding_answer_maps_to_the_canonical_option(monkeypatch):
    monkeypatch.setattr(ov, "HostedProvider", _provider_returning("moderate"))
    result = ov.classify_onboarding_answer("risk_tolerance", "I'd say somewhere in the middle")
    assert result == "Moderate"  # canonical casing, not the LLM's own "moderate"


def test_classify_onboarding_answer_none_when_llm_says_none(monkeypatch):
    monkeypatch.setattr(ov, "HostedProvider", _provider_returning("NONE"))
    result = ov.classify_onboarding_answer("risk_tolerance", "what's the weather like")
    assert result is None


def test_classify_onboarding_answer_none_on_an_unparseable_response(monkeypatch):
    class _GarbageProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return "I refuse to answer that."

    monkeypatch.setattr(ov, "HostedProvider", _GarbageProvider)
    result = ov.classify_onboarding_answer("risk_tolerance", "moderate I guess")
    assert result is None


def test_classify_onboarding_answer_none_when_the_llm_invents_an_option(monkeypatch):
    # Never trust the LLM's own text past matching it back to a KNOWN
    # option -- an invented value must degrade to None, not pass through.
    monkeypatch.setattr(ov, "HostedProvider", _provider_returning("Somewhat Aggressive"))
    result = ov.classify_onboarding_answer("risk_tolerance", "pretty aggressive I'd say")
    assert result is None


def test_classify_onboarding_answer_falls_back_to_none_on_llm_outage(monkeypatch):
    monkeypatch.setattr(ov, "HostedProvider", _RaisingProvider)
    result = ov.classify_onboarding_answer("risk_tolerance", "moderate")
    assert result is None


def test_classify_onboarding_answer_works_for_every_field(monkeypatch):
    monkeypatch.setattr(ov, "HostedProvider", _provider_returning("yes"))
    assert ov.classify_onboarding_answer("interested_in_crypto", "yeah definitely") == "Yes"
    assert ov.classify_onboarding_answer("interested_in_real_estate", "sure") == "Yes"


def test_classify_onboarding_answer_raises_on_an_unknown_field():
    with pytest.raises(KeyError):
        ov.classify_onboarding_answer("not_a_real_field", "moderate")
