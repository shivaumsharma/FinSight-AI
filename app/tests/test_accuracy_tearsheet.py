"""
Tests for app/reporting/accuracy_tearsheet.py (build_accuracy_tearsheet)
and GET /v1/accuracy-tearsheet.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import db, jobs
from app.api import main
from app.api.main import app
from app.core import llm_provider as lp
from app.reporting import accuracy_tearsheet as at


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setattr(lp, "_provider", None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(jobs, "REPORTS_DIR", tmp_path / "reports")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    resp = client.post("/v1/auth/signup", json={"email": "tearsheet@example.com", "password": "tearsheetpassword"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_result(**overrides):
    result = {
        "metric": "12-month forward directional accuracy",
        "n": 1500,
        "model_accuracy_pct": 61.2,
        "model_ci_95": [58.7, 63.6],
        "always_buy_baseline_pct": 52.1,
        "always_buy_ci_95": [49.5, 54.6],
        "beats_baseline": True,
        "summary_line": "61.2% forward-accuracy (N=1500)",
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------- build_accuracy_tearsheet

def test_build_tearsheet_reads_the_canonical_result_file(tmp_path, monkeypatch):
    result_path = tmp_path / "canonical_accuracy_result.json"
    result_path.write_text(json.dumps(_fake_result()))
    monkeypatch.setattr(at, "_RESULT_PATH", result_path)
    monkeypatch.setattr(at.snowflake_accuracy_store, "connect", lambda: None)

    tearsheet = at.build_accuracy_tearsheet()

    assert tearsheet["available"] is True
    assert tearsheet["canonical"]["model_accuracy_pct"] == 61.2
    assert tearsheet["live_breakdowns"] is None


def test_build_tearsheet_degrades_when_the_result_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "_RESULT_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setattr(at.snowflake_accuracy_store, "connect", lambda: None)

    tearsheet = at.build_accuracy_tearsheet()

    assert tearsheet["available"] is False
    assert tearsheet["canonical"] is None


def test_build_tearsheet_includes_live_breakdowns_when_snowflake_is_configured(tmp_path, monkeypatch):
    result_path = tmp_path / "canonical_accuracy_result.json"
    result_path.write_text(json.dumps(_fake_result()))
    monkeypatch.setattr(at, "_RESULT_PATH", result_path)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(at.snowflake_accuracy_store, "connect", lambda: _FakeConn())
    monkeypatch.setattr(
        at.snowflake_accuracy_store,
        "run_validation_queries",
        lambda conn: {"accuracy_trend_by_run_date": [("2025-09-07", 100, 61.0)]},
    )

    tearsheet = at.build_accuracy_tearsheet()

    assert tearsheet["live_breakdowns"] == {"accuracy_trend_by_run_date": [["2025-09-07", 100, 61.0]]}


def test_build_tearsheet_live_breakdowns_none_on_a_query_failure(tmp_path, monkeypatch):
    result_path = tmp_path / "canonical_accuracy_result.json"
    result_path.write_text(json.dumps(_fake_result()))
    monkeypatch.setattr(at, "_RESULT_PATH", result_path)

    class _FakeConn:
        def close(self):
            pass

    def _raise(conn):
        raise RuntimeError("warehouse suspended")

    monkeypatch.setattr(at.snowflake_accuracy_store, "connect", lambda: _FakeConn())
    monkeypatch.setattr(at.snowflake_accuracy_store, "run_validation_queries", _raise)

    tearsheet = at.build_accuracy_tearsheet()

    # A live-data failure must not take down the canonical number too.
    assert tearsheet["available"] is True
    assert tearsheet["live_breakdowns"] is None


# ---------------------------------------------------------------- GET /v1/accuracy-tearsheet

def test_accuracy_tearsheet_endpoint_requires_a_session(client):
    assert client.get("/v1/accuracy-tearsheet").status_code == 401


def test_accuracy_tearsheet_endpoint_returns_the_built_tearsheet(client, monkeypatch, auth_headers):
    monkeypatch.setattr(
        main, "build_accuracy_tearsheet",
        lambda: {"available": True, "canonical": _fake_result(), "live_breakdowns": None},
    )

    resp = client.get("/v1/accuracy-tearsheet", headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["canonical"]["model_accuracy_pct"] == 61.2
