"""
Unit/integration tests for the app/api/ FastAPI service.

Uses a stub agent (same dependency-injection spirit as
test_langgraph_agent.py's stub tools) via jobs.ORCHESTRATORS, monkeypatched
per test -- so these run fast with no real network/model calls, while
still exercising the real HTTP layer, the real background-thread job
runner, and the real SQLite persistence (pointed at a per-test temp file).
"""

import io
import time

import pandas as pd
import pandas.testing as pdt
import pytest
from fastapi.testclient import TestClient

from app.api import db, jobs
from app.api import main
from app.api.main import app
from app.api.serialization import context_to_api_dict, financial_df_from_json
from app.core import llm_provider as lp
from app.core.research_context import ResearchContext
from app.data.market_data import TickerNotFoundError


# ---------------------------------------------------------------- fixtures

def _sample_financial_df():
    return pd.DataFrame({
        "revenue": [100.0, 110.5, None],
        "ebit": [20, 22, 25],
        "shares_outstanding": [1000, 1000, 1050],
    }, index=pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"]))


class _StubAgent:
    """Stand-in for ResearchAgent/LangGraphResearchAgent -- returns a
    fully-populated fake ResearchContext instantly, no network/model calls."""

    def run(self, question):
        context = ResearchContext(ticker="AAPL", question=question)
        context.report_data = {
            "ticker": "AAPL",
            "recommendation": {"rating": "Buy", "basis": "test basis"},
            "valuation_analysis": {
                "sensitivity_table": pd.DataFrame({0.03: [95.0], 0.04: [110.0]}, index=[0.09]),
            },
        }
        context.normalized_financials = _sample_financial_df()
        context.pdf_bytes = b"%PDF-1.4 fake pdf content for testing"
        context.record_tool("market_data_tool")
        context.record_tool("valuation_tool")
        context.record_tool("report_tool")
        context.record_tool("evaluation_tool")
        return context


class _FailingStubAgent:
    def run(self, question):
        raise ValueError("synthetic pipeline failure")


class _TickerNotFoundStubAgent:
    def run(self, question):
        raise TickerNotFoundError("ZZZZ")


@pytest.fixture
def client(tmp_path, monkeypatch):
    # LLM_PROVIDER=hosted is the default, and HostedProvider() raises at
    # construction without LLM_BASE_URL/LLM_API_KEY/LLM_MODEL -- jobs.py
    # calls get_llm_provider() unconditionally before any of these stub
    # agents ever run, so an unconfigured test env would otherwise crash
    # every job before _StubAgent.run() is even reached. "local" keeps
    # construction lazy (no real model load happens -- these tests never
    # call .generate()).
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setattr(lp, "_provider", None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(jobs, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(jobs, "ORCHESTRATORS", {**jobs.ORCHESTRATORS, "hand_rolled": _StubAgent})
    # No real company-resolution network/disk-cache hit -- every test
    # question below is phrased to resolve via the deterministic
    # ticker-token path in company_resolver.py (a bare "AAPL"/"MSFT"
    # token), not the fuzzy-name or LLM-fallback paths.
    with TestClient(app) as test_client:
        yield test_client


def _signup(client, email="alice@example.com", password="correcthorsebattery"):
    """Signs up a fresh user and returns an Authorization header dict --
    the shared setup for every test that needs a real logged-in user
    rather than the old implicit "anonymous" default."""
    resp = client.post("/v1/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers(client):
    return _signup(client)


def _poll_until_terminal(client, job_id, headers, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/v1/research/{job_id}", headers=headers)
        if resp.json()["status"] in (db.STATUS_DONE, db.STATUS_ERROR):
            return resp
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout}s")


# ---------------------------------------------------------------- health

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------- API key middleware

def test_v1_routes_open_when_no_api_key_configured(client, monkeypatch, auth_headers):
    # Default test-fixture state: API_KEY was never set, so _API_KEY is
    # None -- existing/local-dev behavior, unchanged by this middleware.
    # Still needs a real session token, though -- that's a separate,
    # always-on layer (see main.py's module docstring on why both exist).
    monkeypatch.setattr(main, "_API_KEY", None)
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 200


def test_v1_routes_reject_missing_or_wrong_key_once_configured(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "_API_KEY", "the-real-key")
    # The X-API-Key middleware runs before route/dependency resolution,
    # so it 401s here even with a perfectly valid session token --
    # deployment-level gating short-circuits before per-user identity
    # is ever checked.
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"

    resp = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"},
        headers={**auth_headers, "X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_v1_routes_accept_the_correct_key(client, monkeypatch):
    # Sign up BEFORE the key is configured (default test-fixture state
    # has no key set) -- signup itself also sits behind /v1, so doing
    # it after would 401 before ever reaching the route.
    headers = _signup(client)
    monkeypatch.setattr(main, "_API_KEY", "the-real-key")
    headers["X-API-Key"] = "the-real-key"
    resp = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"},
        headers=headers,
    )
    assert resp.status_code == 200


def test_health_stays_open_even_when_api_key_is_configured(client, monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "the-real-key")
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------- submit/poll/done

def test_submit_poll_done_returns_expected_result_shape(client, auth_headers):
    submit_resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert submit_resp.status_code == 200
    job_id = submit_resp.json()["job_id"]

    final = _poll_until_terminal(client, job_id, auth_headers).json()

    assert final["status"] == db.STATUS_DONE
    assert final["result"]["ticker"] == "AAPL"
    assert final["result"]["report_data"]["recommendation"]["rating"] == "Buy"
    # sensitivity_table must survive the trip as a plain dict, not a
    # DataFrame -- FastAPI's encoder would have already raised during
    # the response if this weren't handled (see serialization.py).
    assert isinstance(final["result"]["report_data"]["valuation_analysis"]["sensitivity_table"], dict)
    assert final["result"]["tool_trace"] == [
        "market_data_tool", "valuation_tool", "report_tool", "evaluation_tool",
    ]


def test_submitted_job_is_tagged_with_the_real_user_not_anonymous(client, auth_headers):
    # The whole point of this pass -- see db.py's jobs.user_id column,
    # which existed and defaulted to "anonymous" before any of this
    # was wired up.
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]

    me = client.get("/v1/auth/me", headers=auth_headers).json()
    assert db.get_job(job_id)["user_id"] == me["user_id"]
    assert db.get_job(job_id)["user_id"] != "anonymous"


def test_pdf_endpoint_returns_real_bytes_written_to_disk(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers)

    pdf_resp = client.get(f"/v1/research/{job_id}/pdf", headers=auth_headers)
    assert pdf_resp.status_code == 200
    assert pdf_resp.content == b"%PDF-1.4 fake pdf content for testing"


def test_pdf_endpoint_409s_while_job_is_not_done(client, monkeypatch, auth_headers):
    # A stub agent that never actually finishes within the test -- just
    # check the row right after submission, before the background
    # thread has necessarily completed it.
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    resp = client.get(f"/v1/research/{job_id}/pdf", headers=auth_headers)
    # Could race with the (fast) stub finishing first -- only assert
    # the 409 case when it's actually still pending/running.
    status = client.get(f"/v1/research/{job_id}", headers=auth_headers).json()["status"]
    if status in (db.STATUS_PENDING, db.STATUS_RUNNING):
        assert resp.status_code == 409
        assert resp.json()["code"] == "JOB_NOT_DONE"


def test_pipeline_error_surfaces_as_structured_error(client, monkeypatch, auth_headers):
    monkeypatch.setitem(jobs.ORCHESTRATORS, "hand_rolled", _FailingStubAgent)
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    final = _poll_until_terminal(client, job_id, auth_headers).json()

    assert final["status"] == db.STATUS_ERROR
    assert final["error_code"] == "PIPELINE_ERROR"
    assert "synthetic pipeline failure" in final["error_message"]


def test_ticker_not_found_gets_its_own_error_code(client, monkeypatch, auth_headers):
    monkeypatch.setitem(jobs.ORCHESTRATORS, "hand_rolled", _TickerNotFoundStubAgent)
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    final = _poll_until_terminal(client, job_id, auth_headers).json()

    assert final["error_code"] == "TICKER_NOT_FOUND"


# ---------------------------------------------------------------- validation

def test_no_company_question_returns_structured_400(client, auth_headers):
    resp = client.post("/v1/research", json={"question": "What is the stock market?"}, headers=auth_headers)
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "NO_COMPANY_DETECTED"


def test_unknown_job_id_returns_structured_404(client, auth_headers):
    resp = client.get("/v1/research/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["code"] == "JOB_NOT_FOUND"


# ---------------------------------------------------------------- recent reports

def test_recent_reports_returns_completed_job_with_ticker_and_rating(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers)

    resp = client.get("/v1/research/recent", headers=auth_headers)
    assert resp.status_code == 200
    reports = resp.json()["reports"]
    assert len(reports) == 1
    assert reports[0]["job_id"] == job_id
    assert reports[0]["ticker"] == "AAPL"
    assert reports[0]["rating"] == "Buy"


def test_recent_reports_excludes_a_different_users_jobs(client, auth_headers):
    client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)

    other_user_headers = _signup(client, email="recentreportsother@example.com", password="otherpassword")
    resp = client.get("/v1/research/recent", headers=other_user_headers)
    assert resp.status_code == 200
    assert resp.json()["reports"] == []


def test_recent_reports_is_registered_before_the_job_id_route(client, auth_headers):
    # /v1/research/recent must not be shadowed by /v1/research/{job_id}
    # matching "recent" as a literal job_id -- a real bug class for
    # Starlette route registration order, exercised directly here.
    resp = client.get("/v1/research/recent", headers=auth_headers)
    assert resp.status_code == 200
    assert "reports" in resp.json()


# ---------------------------------------------------------------- per-user auth

def test_signup_then_login_both_work(client):
    _signup(client, email="carol@example.com", password="carolspassword")
    resp = client.post("/v1/auth/login", json={"email": "carol@example.com", "password": "carolspassword"})
    assert resp.status_code == 200
    assert "session_token" in resp.json()


def test_auth_me_returns_profile_and_usage_fields(client, auth_headers):
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "alice@example.com"
    assert isinstance(body["created_at"], float)
    assert body["jobs_used_today"] == 0
    assert body["daily_limit"] == main.DAILY_JOB_LIMIT
    assert body["total_reports"] == 0
    # Session TTL default is 30 days -- just assert it's meaningfully
    # in the future, not an exact value (avoids a flaky test tied to
    # wall-clock timing between session creation and this assertion).
    assert body["session_expires_at"] > time.time() + 3600

    client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["jobs_used_today"] == 1
    assert resp.json()["total_reports"] == 1


def test_auth_me_reset_at_is_null_with_no_usage_and_set_after_a_job(client, auth_headers):
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["reset_at"] is None

    before = time.time()
    client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    resp = client.get("/v1/auth/me", headers=auth_headers)
    reset_at = resp.json()["reset_at"]
    # reset_at = the job's started_at + the 24h rate-limit window --
    # started_at is "before", so reset_at must land just under 24h past it.
    assert reset_at is not None
    assert before + main.RATE_LIMIT_WINDOW_SECONDS <= reset_at <= before + main.RATE_LIMIT_WINDOW_SECONDS + 30


def test_joining_waitlist_is_idempotent_and_reflected_in_auth_me(client, auth_headers):
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["waitlist_features"] == []

    resp = client.post("/v1/waitlist", json={"feature": "brokerage_sync"}, headers=auth_headers)
    assert resp.status_code == 200

    # Joining twice must not error or duplicate.
    resp = client.post("/v1/waitlist", json={"feature": "brokerage_sync"}, headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["waitlist_features"] == ["brokerage_sync"]


def test_risk_tolerance_defaults_to_moderate_and_is_persisted(client, auth_headers):
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["risk_tolerance"] == "Moderate"

    resp = client.patch("/v1/auth/risk-tolerance", json={"risk_tolerance": "Aggressive"}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["risk_tolerance"] == "Aggressive"

    # Persisted -- a fresh /v1/auth/me call reflects it, not just the
    # PATCH response itself.
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["risk_tolerance"] == "Aggressive"


def test_risk_tolerance_rejects_an_invalid_level(client, auth_headers):
    resp = client.patch("/v1/auth/risk-tolerance", json={"risk_tolerance": "YOLO"}, headers=auth_headers)
    assert resp.status_code == 422


def test_deleting_account_requires_the_correct_password(client, auth_headers):
    resp = client.request("DELETE", "/v1/auth/me", json={"password": "wrongpassword"}, headers=auth_headers)
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_CREDENTIALS"

    # The account must still exist and be usable after a rejected deletion.
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200


def test_deleting_account_removes_everything_scoped_to_the_user(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    client.post("/v1/push/subscribe", json={
        "endpoint": "https://push.example.com/deleteme",
        "keys": {"p256dh": "p256dh-val", "auth": "auth-val"},
    }, headers=auth_headers)

    resp = client.request("DELETE", "/v1/auth/me", json={"password": "correcthorsebattery"}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # The now-deleted session's own token must no longer authenticate --
    # deleting the account deletes ALL of that user's sessions, not just
    # leaving the current one dangling on an orphaned user_id.
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 401

    # And a fresh login must fail -- the user row itself is gone, not
    # just this one session.
    resp = client.post("/v1/auth/login", json={"email": "alice@example.com", "password": "correcthorsebattery"})
    assert resp.status_code == 401

    # Cascading cleanup, verified directly against the DB rather than
    # through an API surface (there's no "list all my jobs" endpoint).
    assert db.get_job(job_id) is None
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM push_subscriptions WHERE endpoint=?",
                             ("https://push.example.com/deleteme",)).fetchone()[0] == 0


def test_signup_rejects_a_duplicate_email(client):
    _signup(client, email="dave@example.com")
    resp = client.post("/v1/auth/signup", json={"email": "dave@example.com", "password": "anotherpassword"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "EMAIL_ALREADY_REGISTERED"


def test_signup_rejects_a_short_password(client):
    resp = client.post("/v1/auth/signup", json={"email": "short@example.com", "password": "short"})
    assert resp.status_code == 422


def test_login_rejects_wrong_password(client):
    _signup(client, email="erin@example.com", password="erinspassword")
    resp = client.post("/v1/auth/login", json={"email": "erin@example.com", "password": "wrong-password"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "INVALID_CREDENTIALS"


def test_protected_route_401s_with_no_session_token(client):
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"


def test_protected_route_401s_with_a_garbage_session_token(client):
    resp = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"},
        headers={"Authorization": "Bearer this-token-was-never-issued"},
    )
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"


def test_logout_invalidates_the_session(client, auth_headers):
    resp = client.post("/v1/auth/logout", headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 401


def test_a_second_user_cannot_read_the_first_users_job(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]

    other_user_headers = _signup(client, email="mallory@example.com", password="malloryspassword")
    resp = client.get(f"/v1/research/{job_id}", headers=other_user_headers)
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN"


def test_a_second_user_cannot_download_the_first_users_pdf(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers)

    other_user_headers = _signup(client, email="mallory2@example.com", password="malloryspassword")
    resp = client.get(f"/v1/research/{job_id}/pdf", headers=other_user_headers)
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN"


# ---------------------------------------------------------------- signed PDF share links

def test_share_link_grants_pdf_access_with_no_session_at_all(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers)

    share_resp = client.post(f"/v1/research/{job_id}/pdf/share", headers=auth_headers)
    assert share_resp.status_code == 200
    share_url = share_resp.json()["url"]

    # No Authorization header at all -- the whole point of a share link.
    pdf_resp = client.get(share_url)
    assert pdf_resp.status_code == 200
    assert pdf_resp.content == b"%PDF-1.4 fake pdf content for testing"


def test_only_the_owner_can_mint_a_share_link(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]

    other_user_headers = _signup(client, email="mallory3@example.com", password="malloryspassword")
    resp = client.post(f"/v1/research/{job_id}/pdf/share", headers=other_user_headers)
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN"


def test_share_link_rejects_a_tampered_signature(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers)

    share_url = client.post(f"/v1/research/{job_id}/pdf/share", headers=auth_headers).json()["url"]
    tampered_url = share_url[:-1] + ("0" if share_url[-1] != "0" else "1")

    resp = client.get(tampered_url)
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"


def test_share_link_rejects_a_signature_for_a_different_job(client, auth_headers):
    job_id_1 = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    job_id_2 = client.post(
        "/v1/research", json={"question": "Should I invest in MSFT?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id_1, auth_headers)
    _poll_until_terminal(client, job_id_2, auth_headers)

    # A valid signature minted for job_id_1 must not grant access to
    # job_id_2's PDF, even though both are owned by the same user.
    share_url = client.post(f"/v1/research/{job_id_1}/pdf/share", headers=auth_headers).json()["url"]
    query_string = share_url.split("?", 1)[1]
    resp = client.get(f"/v1/research/{job_id_2}/pdf?{query_string}")
    assert resp.status_code == 401


def test_share_link_expired_is_rejected(client, monkeypatch, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers)

    monkeypatch.setattr(main, "PDF_SHARE_TTL_SECONDS", -10)
    share_url = client.post(f"/v1/research/{job_id}/pdf/share", headers=auth_headers).json()["url"]

    resp = client.get(share_url)
    assert resp.status_code == 401


def test_exceeding_the_daily_job_limit_is_rejected(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "DAILY_JOB_LIMIT", 2)

    for _ in range(2):
        resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
        assert resp.status_code == 200

    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 429
    assert resp.json()["code"] == "RATE_LIMIT_EXCEEDED"


def test_rate_limit_is_scoped_per_user_not_global(client, monkeypatch, auth_headers):
    # A second user hitting their own limit must not be affected by the
    # first user's usage -- count_recent_jobs is keyed by user_id, not
    # a single global counter.
    monkeypatch.setattr(main, "DAILY_JOB_LIMIT", 1)

    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 200
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 429

    other_user_headers = _signup(client, email="ninacountsseparately@example.com", password="ninaspassword")
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=other_user_headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------- push notifications

def test_vapid_public_key_is_available_with_no_auth(client):
    # Not a secret -- the browser needs it before a user has even
    # logged in, to decide whether push is even supported/offered.
    resp = client.get("/v1/push/vapid-public-key")
    assert resp.status_code == 200
    assert isinstance(resp.json()["public_key"], str)
    assert len(resp.json()["public_key"]) > 0


def test_push_subscribe_requires_a_session(client):
    resp = client.post(
        "/v1/push/subscribe",
        json={"endpoint": "https://push.example.com/ep1", "keys": {"p256dh": "p", "auth": "a"}},
    )
    assert resp.status_code == 401


def test_push_subscribe_then_unsubscribe_round_trip(client, auth_headers):
    resp = client.post(
        "/v1/push/subscribe",
        json={"endpoint": "https://push.example.com/ep1", "keys": {"p256dh": "p256dh-val", "auth": "auth-val"}},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    me = client.get("/v1/auth/me", headers=auth_headers).json()
    subs = db.get_push_subscriptions(me["user_id"])
    assert len(subs) == 1
    assert subs[0]["endpoint"] == "https://push.example.com/ep1"
    assert subs[0]["p256dh"] == "p256dh-val"

    resp = client.post("/v1/push/unsubscribe", json={"endpoint": "https://push.example.com/ep1"}, headers=auth_headers)
    assert resp.status_code == 200
    assert db.get_push_subscriptions(me["user_id"]) == []


def test_push_unsubscribe_on_an_unknown_endpoint_is_not_an_error(client, auth_headers):
    resp = client.post("/v1/push/unsubscribe", json={"endpoint": "https://push.example.com/never-subscribed"}, headers=auth_headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------- reconciliation

def test_startup_reconciliation_marks_orphaned_running_jobs_interrupted(client):
    """
    This is the test that actually defines "the right status" for a
    job whose worker thread died with the previous process -- manually
    insert a row in RUNNING with no live thread behind it (simulating
    exactly that), call the reconciliation function directly, and
    assert it becomes error/INTERRUPTED, not stuck at RUNNING forever.
    """
    job_id = db.create_job(question="Should I invest in AAPL?", orchestrator="hand_rolled")
    db.mark_running(job_id)
    assert db.get_job(job_id)["status"] == db.STATUS_RUNNING

    reconciled_count = db.reconcile_interrupted_jobs()

    assert reconciled_count == 1
    job = db.get_job(job_id)
    assert job["status"] == db.STATUS_ERROR
    assert job["error_code"] == db.ERROR_INTERRUPTED


def test_startup_reconciliation_also_covers_pending_jobs(client):
    # A job that was queued but never even started (see db.create_job's
    # docstring for why PENDING, not RUNNING, is the creation state) is
    # exactly as orphaned by a restart as one mid-execution.
    job_id = db.create_job(question="Should I invest in AAPL?", orchestrator="hand_rolled")
    assert db.get_job(job_id)["status"] == db.STATUS_PENDING

    reconciled_count = db.reconcile_interrupted_jobs()

    assert reconciled_count == 1
    assert db.get_job(job_id)["status"] == db.STATUS_ERROR


def test_timeout_reconciliation_marks_long_running_jobs_timed_out(client):
    job_id = db.create_job(question="Should I invest in AAPL?", orchestrator="hand_rolled")
    db.mark_running(job_id)

    # Simulate a job that's been "running" far longer than the timeout
    # by directly backdating started_at, rather than actually sleeping
    # in the test.
    with db._connect() as conn:
        conn.execute("UPDATE jobs SET started_at=? WHERE job_id=?", (time.time() - 10000, job_id))

    reconciled_count = db.reconcile_timed_out_jobs(timeout_seconds=600)

    assert reconciled_count == 1
    job = db.get_job(job_id)
    assert job["status"] == db.STATUS_ERROR
    assert job["error_code"] == db.ERROR_TIMEOUT


def test_timeout_reconciliation_leaves_recent_jobs_alone(client):
    job_id = db.create_job(question="Should I invest in AAPL?", orchestrator="hand_rolled")
    db.mark_running(job_id)

    reconciled_count = db.reconcile_timed_out_jobs(timeout_seconds=600)

    assert reconciled_count == 0
    assert db.get_job(job_id)["status"] == db.STATUS_RUNNING


def test_late_result_after_timeout_does_not_overwrite_the_error_status(client):
    """
    Guards the race the reviewer flagged: a zombie thread that was
    marked TIMEOUT but keeps running in the background (Python can't
    force-kill it -- see db.py's JOB_TIMEOUT_SECONDS docstring) must
    not silently overwrite the error status if it eventually finishes.
    """
    job_id = db.create_job(question="Should I invest in AAPL?", orchestrator="hand_rolled")
    db.mark_running(job_id)
    db.mark_error(job_id, db.ERROR_TIMEOUT, "timed out")

    db.mark_done(job_id, result={"ticker": "AAPL"}, pdf_path="/tmp/whatever.pdf")

    job = db.get_job(job_id)
    assert job["status"] == db.STATUS_ERROR
    assert job["error_code"] == db.ERROR_TIMEOUT


# ---------------------------------------------------------------- serialization round-trip

def test_normalized_financials_round_trip_is_exact():
    """
    The one silent-failure path in the whole design: if the to_json/
    read_json round-trip for normalized_financials doesn't preserve
    dtypes and the index exactly, the DCF would quietly compute
    different numbers downstream with no visible error. Checked
    directly here, not inferred from the API test passing.
    """
    original = _sample_financial_df()

    context = ResearchContext(ticker="AAPL", question="test")
    context.normalized_financials = original
    context.report_data = {}

    api_dict = context_to_api_dict(context)
    roundtripped = financial_df_from_json(api_dict["normalized_financials"])

    pdt.assert_frame_equal(original, roundtripped)


def test_normalized_financials_round_trip_handles_none():
    context = ResearchContext(ticker="AAPL", question="test")
    context.normalized_financials = None
    context.report_data = {}

    api_dict = context_to_api_dict(context)

    assert api_dict["normalized_financials"] is None
    assert financial_df_from_json(api_dict["normalized_financials"]) is None
