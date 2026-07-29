"""
Unit tests for app/core/llm_provider.py (the pluggable LLMProvider
interface) and its downstream wiring into app/planner/llm_planner.py,
app/core/company_resolver.py, and app/api/jobs.py.

No real network/model calls anywhere -- requests.post is mocked for
every HostedProvider test, and time.sleep is patched to a no-op so the
backoff tests don't actually wait.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core import llm_provider as lp
from app.core.llm_provider import (
    HostedProvider, LLMProviderError, LocalLlamaProvider,
    get_llm_provider, is_local_provider,
)


def _mock_response(status_code, json_data=None, text="", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _chat_response(content="hello", usage=None):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# ---------------------------------------------------------------- provider selection

def test_get_llm_provider_defaults_to_hosted(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(lp, "_provider", None)
    assert isinstance(get_llm_provider(), HostedProvider)


def test_get_llm_provider_selects_local(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setattr(lp, "_provider", None)
    assert isinstance(get_llm_provider(), LocalLlamaProvider)


def test_is_local_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert is_local_provider() is False
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    assert is_local_provider() is False
    monkeypatch.setenv("LLM_PROVIDER", "local")
    assert is_local_provider() is True


# ---------------------------------------------------------------- LLM call JSONL logging

def _get_logged_provider(monkeypatch, tmp_path, log_dir=None):
    """get_llm_provider() with a real HostedProvider, logging enabled
    and redirected to a tmp dir, isolated from whatever real LLM_LOG_DIR
    app/core/paths.py computed at import time."""
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.setenv("LLM_BASE_URL", "https://x.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(lp, "_provider", None)
    monkeypatch.setattr(lp, "_LOG_LLM_CALLS_ENABLED", True)
    monkeypatch.setattr(lp, "LLM_LOG_DIR", log_dir or tmp_path)
    return get_llm_provider()


def _read_jsonl_records(log_dir):
    import json as _json
    files = list(log_dir.glob("llm_calls_*.jsonl"))
    assert len(files) == 1, f"expected exactly one log file, found {files}"
    with open(files[0], encoding="utf-8") as f:
        return [_json.loads(line) for line in f if line.strip()]


def test_logging_writes_expected_fields(monkeypatch, tmp_path):
    provider = _get_logged_provider(monkeypatch, tmp_path)
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response("hi there"))):
        result = provider.generate("the real prompt", max_new_tokens=50)

    assert result == "hi there"
    records = _read_jsonl_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["provider"] == "HostedProvider"
    assert record["model"] == "test-model"
    assert record["prompt"] == "the real prompt"
    assert record["response"] == "hi there"
    assert record["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert record["error"] is None
    assert record["latency_seconds"] >= 0
    assert "timestamp" in record


def test_logging_usage_is_per_call_delta_not_cumulative(monkeypatch, tmp_path):
    provider = _get_logged_provider(monkeypatch, tmp_path)
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response())):
        provider.generate("p1")
        provider.generate("p2")

    records = _read_jsonl_records(tmp_path)
    assert len(records) == 2
    # get_last_usage() accumulates (see test_hosted_usage_accumulates_across_calls) --
    # each logged record must still show THIS call's usage, not the running total.
    assert records[0]["usage"]["total_tokens"] == 15
    assert records[1]["usage"]["total_tokens"] == 15


def test_logging_records_errors_and_still_raises(monkeypatch, tmp_path):
    provider = _get_logged_provider(monkeypatch, tmp_path)
    with patch("app.core.llm_provider.requests.post",
               return_value=_mock_response(401, text="bad api key")), \
         patch("app.core.llm_provider.time.sleep"):
        with pytest.raises(LLMProviderError):
            provider.generate("a prompt that will fail")

    records = _read_jsonl_records(tmp_path)
    assert len(records) == 1
    assert records[0]["response"] is None
    assert records[0]["error"] is not None
    assert "401" in records[0]["error"]


def test_logging_disabled_via_flag_writes_nothing(monkeypatch, tmp_path):
    provider = _get_logged_provider(monkeypatch, tmp_path)
    monkeypatch.setattr(lp, "_LOG_LLM_CALLS_ENABLED", False)
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response())):
        provider.generate("p1")
    assert list(tmp_path.glob("llm_calls_*.jsonl")) == []


def test_logging_failure_does_not_break_the_real_call(monkeypatch, tmp_path):
    # LLM_LOG_DIR pointed at something that can't be created (a file,
    # not a directory) -- mkdir() must raise, and generate() must still
    # succeed and return the real result regardless.
    unwritable = tmp_path / "not_a_directory"
    unwritable.write_text("occupied")
    provider = _get_logged_provider(monkeypatch, tmp_path, log_dir=unwritable)
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response("hi there"))):
        result = provider.generate("p1")
    assert result == "hi there"


# ---------------------------------------------------------------- HostedProvider config

def test_hosted_provider_requires_all_three_env_vars():
    with pytest.raises(LLMProviderError):
        HostedProvider(base_url=None, api_key=None, model=None)


def test_hosted_provider_raises_at_construction_not_first_call(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(LLMProviderError):
        HostedProvider()


def test_hosted_provider_accepts_constructor_args_over_env():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m")
    assert provider.base_url == "https://x.test/v1"


# ---------------------------------------------------------------- HostedProvider.generate happy path

def test_hosted_generate_returns_content_and_captures_usage():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m")
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response("hi there"))):
        result = provider.generate("prompt", max_new_tokens=50)
    assert result == "hi there"
    assert provider.get_last_usage() == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_hosted_usage_accumulates_across_calls():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m")
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response())):
        provider.generate("p1")
        provider.generate("p2")
    assert provider.get_last_usage()["total_tokens"] == 30


def test_reset_usage_zeroes_out():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m")
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response())):
        provider.generate("p1")
    assert provider.get_last_usage() is not None
    provider.reset_usage()
    assert provider.get_last_usage() is None


# ---------------------------------------------------------------- retry/backoff/timeout

def test_429_without_retry_after_uses_exponential_backoff():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m", max_retries=2)
    responses = [_mock_response(429, text="rate limited")] * 2 + [_mock_response(200, _chat_response("ok"))]
    with patch("app.core.llm_provider.requests.post", side_effect=responses) as mock_post, \
         patch("app.core.llm_provider.time.sleep") as mock_sleep:
        result = provider.generate("prompt")
    assert result == "ok"
    assert mock_post.call_count == 3
    # Exponential: 1s then 2s (attempt 0 and 1), no Retry-After header present.
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


def test_429_with_retry_after_header_is_honored():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m", max_retries=2)
    responses = [
        _mock_response(429, text="rate limited", headers={"Retry-After": "7"}),
        _mock_response(200, _chat_response("ok")),
    ]
    with patch("app.core.llm_provider.requests.post", side_effect=responses), \
         patch("app.core.llm_provider.time.sleep") as mock_sleep:
        result = provider.generate("prompt")
    assert result == "ok"
    mock_sleep.assert_called_once_with(7.0)


def test_retry_after_non_numeric_falls_back_to_exponential():
    # Some providers send an HTTP-date form instead of plain seconds --
    # must not crash, must fall back to the fixed schedule.
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m", max_retries=2)
    responses = [
        _mock_response(429, text="rate limited", headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        _mock_response(200, _chat_response("ok")),
    ]
    with patch("app.core.llm_provider.requests.post", side_effect=responses), \
         patch("app.core.llm_provider.time.sleep") as mock_sleep:
        result = provider.generate("prompt")
    assert result == "ok"
    mock_sleep.assert_called_once_with(1.0)


def test_persistent_500_raises_after_exhausting_retries():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m", max_retries=2)
    responses = [_mock_response(500, text="server error")] * 3
    with patch("app.core.llm_provider.requests.post", side_effect=responses) as mock_post, \
         patch("app.core.llm_provider.time.sleep"):
        with pytest.raises(LLMProviderError):
            provider.generate("prompt")
    assert mock_post.call_count == 3  # initial attempt + 2 retries


def test_non_retryable_4xx_fails_on_first_attempt_no_retry():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m", max_retries=3)
    with patch("app.core.llm_provider.requests.post",
               return_value=_mock_response(401, text="bad api key")) as mock_post, \
         patch("app.core.llm_provider.time.sleep") as mock_sleep:
        with pytest.raises(LLMProviderError):
            provider.generate("prompt")
    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_network_error_retries_then_raises():
    import requests as requests_module
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m", max_retries=1)
    with patch("app.core.llm_provider.requests.post",
               side_effect=requests_module.exceptions.ConnectionError("refused")), \
         patch("app.core.llm_provider.time.sleep"):
        with pytest.raises(LLMProviderError):
            provider.generate("prompt")


def test_generate_passes_configured_timeout():
    provider = HostedProvider(base_url="https://x.test/v1", api_key="k", model="m", timeout=15)
    with patch("app.core.llm_provider.requests.post", return_value=_mock_response(200, _chat_response())) as mock_post:
        provider.generate("prompt")
    assert mock_post.call_args.kwargs["timeout"] == 15


# ---------------------------------------------------------------- interface swap at real call sites

class _StubProvider:
    """LLMProvider-shaped stub -- not a subclass, just duck-typed, to
    prove callers only depend on the .generate(prompt, max_new_tokens)
    contract, not on any local-specific behavior."""

    def __init__(self, response="stub response"):
        self.response = response
        self.calls = []

    def generate(self, prompt, max_new_tokens=700):
        self.calls.append((prompt, max_new_tokens))
        return self.response

    def get_last_usage(self):
        return {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    def reset_usage(self):
        pass


def test_llm_planner_uses_injected_provider():
    from app.planner.llm_planner import LLMPlanner
    stub = _StubProvider('{"tools": ["rag_tool"]}')
    planner = LLMPlanner(generator=stub)
    plan = planner.create_plan("What did management say about AI?")
    assert stub.calls  # the stub was actually invoked
    assert "rag_tool" in plan


def test_company_resolver_llm_fallback_uses_get_llm_provider(monkeypatch):
    from app.core import company_resolver

    stub = _StubProvider("Apple")
    monkeypatch.setattr(lp, "get_llm_provider", lambda: stub)

    result = company_resolver._llm_propose_company_name("is apple a good stock to buy")

    assert stub.calls
    assert result == "Apple"


def test_narrative_builder_uses_get_llm_provider(monkeypatch):
    from app.reporting import narrative_builder

    raw_output = "\n".join(f"# {s}\ntext for {s}" for s in narrative_builder.NARRATIVE_SECTIONS)
    stub = _StubProvider(raw_output)
    monkeypatch.setattr(narrative_builder, "get_llm_provider", lambda: stub)
    monkeypatch.setattr(narrative_builder, "cache_get", lambda key: None)
    monkeypatch.setattr(narrative_builder, "cache_set", lambda *a, **k: None)
    monkeypatch.setattr(narrative_builder, "log_narrative_call", lambda **k: None)

    report_data = {
        "ticker": "AAPL",
        "company_overview": {
            "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
            "business_summary": "Apple designs and sells consumer electronics.",
        },
        "growth_analysis": {
            "Revenue Growth (%)": 5.0, "Revenue Trend": "Stable",
            "FCF Growth (%)": 3.0, "FCF Trend": "Stable",
        },
        "valuation_analysis": {"Intrinsic Value (per share)": 150.0, "Current Price": 170.0, "Upside (%)": -11.8},
        "market_earnings_snapshot": {
            "current_price": 170.0, "sentiment_label": "Neutral", "sentiment_confidence": "60%",
            "news_sentiment_label": "Neutral", "news_sentiment_confidence": "60%",
            "next_earnings_date": None,
        },
        "recommendation": {"rating": "Hold", "basis": "test basis"},
    }

    class _FakeContext:
        ticker = "AAPL"
        question = "Should I invest in Apple?"
        news_selected = []
        news_articles = []
        research_summary = ""

    sections = narrative_builder.build_narrative_sections(_FakeContext(), report_data)

    assert stub.calls
    assert sections["Executive Summary"].startswith("text for Executive Summary")


# ---------------------------------------------------------------- app/api/jobs.py wiring

def test_resolve_max_workers_local_ignores_max_concurrent_jobs(monkeypatch):
    from app.api import jobs
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "99")
    assert jobs._resolve_max_workers() == 1


def test_resolve_max_workers_hosted_defaults_to_one(monkeypatch):
    from app.api import jobs
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.delenv("MAX_CONCURRENT_JOBS", raising=False)
    assert jobs._resolve_max_workers() == 1


def test_resolve_max_workers_hosted_honors_explicit_opt_in(monkeypatch):
    from app.api import jobs
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "4")
    assert jobs._resolve_max_workers() == 4


def test_run_job_maps_llm_provider_error_to_structured_code(tmp_path, monkeypatch):
    from app.api import db, jobs

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    db.init_db()

    class _RaisingAgent:
        def run(self, question):
            raise LLMProviderError("hosted provider unreachable")

    job_id = db.create_job(question="Should I invest in AAPL?", orchestrator="hand_rolled")
    jobs._run_job(job_id, "Should I invest in AAPL?", "hand_rolled", agent_factory=_RaisingAgent)

    job = db.get_job(job_id)
    assert job["status"] == db.STATUS_ERROR
    assert job["error_code"] == "LLM_PROVIDER_ERROR"


def test_run_job_stores_llm_usage_in_result(tmp_path, monkeypatch):
    from app.api import db, jobs
    from app.core.research_context import ResearchContext

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    db.init_db()

    stub_provider = _StubProvider()
    monkeypatch.setattr(jobs, "get_llm_provider", lambda: stub_provider)

    class _StubAgent:
        def run(self, question):
            context = ResearchContext(ticker="AAPL", question=question)
            context.report_data = {"recommendation": {"rating": "Hold"}}
            return context

    job_id = db.create_job(question="Should I invest in AAPL?", orchestrator="hand_rolled")
    jobs._run_job(job_id, "Should I invest in AAPL?", "hand_rolled", agent_factory=_StubAgent)

    job = db.get_job(job_id)
    assert job["status"] == db.STATUS_DONE
    assert job["result"]["llm_usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
