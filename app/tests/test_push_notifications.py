"""
Unit tests for app/api/jobs.py's _notify_job_complete and its wiring
into _run_job's outermost finally block.

pywebpush.webpush is never called for real here (mocked via
app.api.auth.send_push_notification) -- no real push service call, no
network. Runs against a per-test temp SQLite DB (same pattern as
test_auth.py's temp_db fixture).
"""

import time

import pytest

from app.api import auth, db, jobs
from app.core import llm_provider as lp
from app.core.llm_provider import LLMProviderError
from app.core.research_context import ResearchContext
from app.data.market_data import TickerNotFoundError


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    # _run_job (jobs.py) unconditionally calls get_llm_provider() before
    # running any agent, real or stubbed -- "local" keeps construction
    # lazy (no real model load, these tests never call .generate())
    # instead of "hosted", which raises at construction without real
    # LLM_BASE_URL/LLM_API_KEY/LLM_MODEL. Same fixture-isolation pattern
    # as test_api.py's own `client` fixture, for the same reason.
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setattr(lp, "_provider", None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


@pytest.fixture
def user_with_subscription(temp_db):
    user_id = temp_db.create_user("push@example.com", "hash", "salt")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh-val", "auth-val")
    return user_id


@pytest.fixture
def sent_notifications(monkeypatch):
    """Records every (endpoint, title, body, status) send_push_notification
    is called with, instead of making a real push service call."""
    calls = []

    def fake_send(subscription, title, body, status):
        calls.append((subscription["endpoint"], title, body, status))

    monkeypatch.setattr(auth, "send_push_notification", fake_send)
    return calls


class _DoneStubAgent:
    def run(self, question):
        context = ResearchContext(ticker="AAPL", question=question)
        context.report_data = {"ticker": "AAPL", "recommendation": {"rating": "Buy"}}
        context.normalized_financials = None
        context.pdf_bytes = None
        return context


class _FailingStubAgent:
    def run(self, question):
        raise ValueError("synthetic pipeline failure")


class _TickerNotFoundStubAgent:
    def run(self, question):
        raise TickerNotFoundError("ZZZZ")


class _LLMProviderErrorStubAgent:
    def run(self, question):
        raise LLMProviderError("hosted provider misconfigured")


# ---------------------------------------------------------------- _notify_job_complete direct tests

def test_notify_sends_ticker_and_rating_on_success(temp_db, user_with_subscription, sent_notifications):
    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": "AAPL", "report_data": {"recommendation": {"rating": "Buy"}}}, None)

    jobs._notify_job_complete(job_id)

    assert len(sent_notifications) == 1
    endpoint, title, body, status = sent_notifications[0]
    assert endpoint == "https://push.example.com/ep1"
    assert "AAPL" in title
    assert "Buy" in body
    assert status == db.STATUS_DONE


def test_notify_sends_a_failure_message_on_error(temp_db, user_with_subscription, sent_notifications):
    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)
    temp_db.mark_running(job_id)
    temp_db.mark_error(job_id, "PIPELINE_ERROR", "synthetic pipeline failure")

    jobs._notify_job_complete(job_id)

    assert len(sent_notifications) == 1
    _, title, body, status = sent_notifications[0]
    assert "failed" in title.lower()
    assert status == db.STATUS_ERROR


def test_notify_skips_users_with_no_subscription(temp_db, sent_notifications):
    user_id = temp_db.create_user("nosubs@example.com", "hash", "salt")
    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_id)
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": "AAPL"}, None)

    jobs._notify_job_complete(job_id)

    assert sent_notifications == []


def test_notify_skips_anonymous_jobs(temp_db, sent_notifications):
    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled")  # default user_id="anonymous"
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": "AAPL"}, None)

    jobs._notify_job_complete(job_id)

    assert sent_notifications == []


def test_notify_sends_to_every_subscription_the_user_holds(temp_db, user_with_subscription, sent_notifications):
    temp_db.add_push_subscription(user_with_subscription, "https://push.example.com/ep2", "p256dh-2", "auth-2")
    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": "AAPL"}, None)

    jobs._notify_job_complete(job_id)

    endpoints = {c[0] for c in sent_notifications}
    assert endpoints == {"https://push.example.com/ep1", "https://push.example.com/ep2"}


def test_notify_prunes_a_subscription_the_push_service_reports_dead(temp_db, user_with_subscription, monkeypatch):
    from pywebpush import WebPushException

    class _FakeResponse:
        status_code = 410

    def fake_send(subscription, title, body, status):
        raise WebPushException("gone", response=_FakeResponse())

    monkeypatch.setattr(auth, "send_push_notification", fake_send)

    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": "AAPL"}, None)

    jobs._notify_job_complete(job_id)

    assert temp_db.get_push_subscriptions(user_with_subscription) == []


def test_notify_does_not_prune_on_a_transient_send_failure(temp_db, user_with_subscription, monkeypatch):
    def fake_send(subscription, title, body, status):
        raise RuntimeError("network blip")

    monkeypatch.setattr(auth, "send_push_notification", fake_send)

    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": "AAPL"}, None)

    jobs._notify_job_complete(job_id)  # must not raise

    assert len(temp_db.get_push_subscriptions(user_with_subscription)) == 1


def test_notify_failing_entirely_does_not_raise(temp_db, monkeypatch):
    # A completely malformed/missing job row -- _notify_job_complete's
    # own outermost try/except must swallow this, matching the
    # "notification failure never propagates" guarantee.
    monkeypatch.setattr(db, "get_job", lambda job_id: (_ for _ in ()).throw(RuntimeError("db exploded")))
    jobs._notify_job_complete("nonexistent-job-id")  # must not raise


# ---------------------------------------------------------------- wired into _run_job

def test_run_job_notifies_exactly_once_on_success(temp_db, user_with_subscription, sent_notifications):
    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)
    jobs._run_job(job_id, "q", "hand_rolled", agent_factory=_DoneStubAgent)

    assert len(sent_notifications) == 1
    assert temp_db.get_job(job_id)["status"] == db.STATUS_DONE


@pytest.mark.parametrize("stub_agent,expected_error_code", [
    (_FailingStubAgent, "PIPELINE_ERROR"),
    (_TickerNotFoundStubAgent, "TICKER_NOT_FOUND"),
    (_LLMProviderErrorStubAgent, "LLM_PROVIDER_ERROR"),
])
def test_run_job_notifies_exactly_once_on_every_error_path(
    temp_db, user_with_subscription, sent_notifications, stub_agent, expected_error_code
):
    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)
    jobs._run_job(job_id, "q", "hand_rolled", agent_factory=stub_agent)

    assert len(sent_notifications) == 1
    job = temp_db.get_job(job_id)
    assert job["status"] == db.STATUS_ERROR
    assert job["error_code"] == expected_error_code


def test_a_broken_notifier_never_changes_an_already_recorded_job_status(temp_db, user_with_subscription, monkeypatch):
    # The guarantee this whole feature is built around: whether a user
    # got notified must never be why a job's own recorded outcome
    # changes. Breaks _notify_job_complete itself (not just the send
    # call inside it) to prove the finally block's failure mode too.
    monkeypatch.setattr(jobs, "_notify_job_complete", lambda job_id: (_ for _ in ()).throw(RuntimeError("notifier is broken")))

    job_id = temp_db.create_job(question="q", orchestrator="hand_rolled", user_id=user_with_subscription)

    with pytest.raises(RuntimeError):
        jobs._run_job(job_id, "q", "hand_rolled", agent_factory=_DoneStubAgent)

    # The job itself completed successfully before the notifier broke --
    # that recorded outcome must survive the notifier's own crash.
    assert temp_db.get_job(job_id)["status"] == db.STATUS_DONE
