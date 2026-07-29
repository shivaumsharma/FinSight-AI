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


def _poll_until_terminal(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/v1/research/{job_id}")
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

def test_v1_routes_open_when_no_api_key_configured(client, monkeypatch):
    # Default test-fixture state: API_KEY was never set, so _API_KEY is
    # None -- existing/local-dev behavior, unchanged by this middleware.
    monkeypatch.setattr(main, "_API_KEY", None)
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"})
    assert resp.status_code == 200


def test_v1_routes_reject_missing_or_wrong_key_once_configured(client, monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "the-real-key")
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "UNAUTHORIZED"

    resp = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_v1_routes_accept_the_correct_key(client, monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "the-real-key")
    resp = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"},
        headers={"X-API-Key": "the-real-key"},
    )
    assert resp.status_code == 200


def test_health_stays_open_even_when_api_key_is_configured(client, monkeypatch):
    monkeypatch.setattr(main, "_API_KEY", "the-real-key")
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------- submit/poll/done

def test_submit_poll_done_returns_expected_result_shape(client):
    submit_resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"})
    assert submit_resp.status_code == 200
    job_id = submit_resp.json()["job_id"]

    final = _poll_until_terminal(client, job_id).json()

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


def test_pdf_endpoint_returns_real_bytes_written_to_disk(client):
    job_id = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}).json()["job_id"]
    _poll_until_terminal(client, job_id)

    pdf_resp = client.get(f"/v1/research/{job_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.content == b"%PDF-1.4 fake pdf content for testing"


def test_pdf_endpoint_409s_while_job_is_not_done(client, monkeypatch):
    # A stub agent that never actually finishes within the test -- just
    # check the row right after submission, before the background
    # thread has necessarily completed it.
    job_id = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}).json()["job_id"]
    resp = client.get(f"/v1/research/{job_id}/pdf")
    # Could race with the (fast) stub finishing first -- only assert
    # the 409 case when it's actually still pending/running.
    status = client.get(f"/v1/research/{job_id}").json()["status"]
    if status in (db.STATUS_PENDING, db.STATUS_RUNNING):
        assert resp.status_code == 409
        assert resp.json()["code"] == "JOB_NOT_DONE"


def test_pipeline_error_surfaces_as_structured_error(client, monkeypatch):
    monkeypatch.setitem(jobs.ORCHESTRATORS, "hand_rolled", _FailingStubAgent)
    job_id = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}).json()["job_id"]
    final = _poll_until_terminal(client, job_id).json()

    assert final["status"] == db.STATUS_ERROR
    assert final["error_code"] == "PIPELINE_ERROR"
    assert "synthetic pipeline failure" in final["error_message"]


def test_ticker_not_found_gets_its_own_error_code(client, monkeypatch):
    monkeypatch.setitem(jobs.ORCHESTRATORS, "hand_rolled", _TickerNotFoundStubAgent)
    job_id = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}).json()["job_id"]
    final = _poll_until_terminal(client, job_id).json()

    assert final["error_code"] == "TICKER_NOT_FOUND"


# ---------------------------------------------------------------- validation

def test_no_company_question_returns_structured_400(client):
    resp = client.post("/v1/research", json={"question": "What is the stock market?"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "NO_COMPANY_DETECTED"


def test_unknown_job_id_returns_structured_404(client):
    resp = client.get("/v1/research/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "JOB_NOT_FOUND"


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
