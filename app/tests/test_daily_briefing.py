"""
Tests for Feature 5 (daily portfolio briefing):
- app/reasoning/daily_briefing.py's build_briefing / sweep_daily_briefings
- chat_router.py's INTENT_DAILY_BRIEFING wiring
- POST /v1/internal/sweep/daily-briefings

Real sqlite temp DB throughout; HostedProvider and
auth.send_push_notification are the only things monkeypatched -- never
real network/push calls.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import auth, db, jobs, main
from app.api.main import app
from app.core import llm_provider as lp
from app.core.llm_provider import LLMProviderError
from app.reasoning import chat_router as cr
from app.reasoning import daily_briefing as db_briefing


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


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
    resp = client.post("/v1/auth/signup", json={"email": "briefing@example.com", "password": "briefingpassword"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


class _FakeProvider:
    """Echoes the prompt's own BRIEFING DATA section back verbatim
    (prefixed) instead of doing real synthesis -- lets assertions check
    that the raw grounding data actually reached the prompt, without
    depending on real LLM wording."""

    def __init__(self, model=None, **kwargs):
        pass

    def generate(self, prompt, max_new_tokens=150):
        return "Synthesized: " + prompt.split("BRIEFING DATA:\n")[1]


class _RaisingProvider:
    def __init__(self, model=None, **kwargs):
        pass

    def generate(self, prompt, max_new_tokens=150):
        raise LLMProviderError("simulated outage")


def _rate(temp_db, user_id, ticker, rating):
    job_id = temp_db.create_job(question=f"research {ticker}", orchestrator="hand_rolled", user_id=user_id)
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": ticker, "report_data": {"recommendation": {"rating": rating}}}, None)


# ---------------------------------------------------------------- build_briefing

def test_build_briefing_reports_no_holdings(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(db_briefing, "HostedProvider", _FakeProvider)

    briefing = db_briefing.build_briefing(user_id)

    assert "No holdings in the portfolio yet" in briefing["text"]
    assert briefing["ticker"] is None


def test_build_briefing_includes_portfolio_pnl(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "AAPL", "BUY", 4, 150.0, "USD")
    monkeypatch.setattr("app.reporting.portfolio_summary.get_quote", lambda t: {"price": 200.0, "change_pct": 1.0, "previous_close": 198.0, "currency": "USD"})
    monkeypatch.setattr(db_briefing, "HostedProvider", _FakeProvider)

    briefing = db_briefing.build_briefing(user_id)

    assert "1 holding" in briefing["text"]
    assert "800.00" in briefing["text"]  # 4 * $200


def test_build_briefing_includes_a_single_rating_change_and_sets_ticker(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "TCS")
    _rate(temp_db, user_id, "TCS", "Sell")
    db.set_rating_alert_seen(user_id, "TCS", "Sell")  # simulates Feature 2 already having proposed it
    monkeypatch.setattr(db_briefing, "HostedProvider", _FakeProvider)

    briefing = db_briefing.build_briefing(user_id)

    assert "TCS moved to Sell" in briefing["text"]
    assert briefing["ticker"] == "TCS"


def test_build_briefing_ticker_is_none_with_multiple_rating_changes(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    db.set_rating_alert_seen(user_id, "TCS", "Sell")
    db.set_rating_alert_seen(user_id, "INFY", "Buy")
    monkeypatch.setattr(db_briefing, "HostedProvider", _FakeProvider)

    briefing = db_briefing.build_briefing(user_id)

    assert "TCS moved to Sell" in briefing["text"]
    assert "INFY moved to Buy" in briefing["text"]
    assert briefing["ticker"] is None


def test_build_briefing_only_mentions_changes_since_the_last_briefing(temp_db, monkeypatch):
    import time

    user_id = temp_db.create_user("a@example.com", "h", "s")
    db.set_rating_alert_seen(user_id, "OLD", "Hold")  # already-known-about, before the cursor
    db.set_last_briefing_at(user_id, time.time())
    db.set_rating_alert_seen(user_id, "NEW", "Sell")  # after the cursor
    monkeypatch.setattr(db_briefing, "HostedProvider", _FakeProvider)

    briefing = db_briefing.build_briefing(user_id)

    assert "NEW moved to Sell" in briefing["text"]
    assert "OLD moved to Hold" not in briefing["text"]


def test_build_briefing_includes_upcoming_events(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "AAPL")
    monkeypatch.setattr(
        "app.reporting.corporate_actions_feed.get_corporate_actions",
        lambda t: {"next_earnings_date": "2026-09-15", "next_ex_dividend_date": None, "last_dividend_amount": None, "last_split": None},
    )
    monkeypatch.setattr("app.reporting.corporate_actions_feed.get_company_name", lambda t: "Apple Inc")
    monkeypatch.setattr(db_briefing, "HostedProvider", _FakeProvider)

    briefing = db_briefing.build_briefing(user_id)

    assert "AAPL earnings on 2026-09-15" in briefing["text"]


def test_build_briefing_falls_back_to_raw_text_on_llm_outage(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    db.set_rating_alert_seen(user_id, "TCS", "Sell")
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    briefing = db_briefing.build_briefing(user_id)

    assert "No holdings in the portfolio yet" in briefing["text"]
    assert "TCS moved to Sell" in briefing["text"]
    assert briefing["ticker"] == "TCS"


# ---------------------------------------------------------------- sweep_daily_briefings

def test_sweep_sends_to_a_never_briefed_user(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)  # exercise the fallback path, no real LLM

    sent = db_briefing.sweep_daily_briefings()

    assert sent == 1
    assert temp_db.get_user_by_id(user_id)["last_briefing_at"] is not None
    messages = temp_db.list_chat_messages(user_id)
    assert len(messages) == 1
    assert messages[0]["intent"] == "daily_briefing"


def test_sweep_skips_a_user_briefed_recently(temp_db, monkeypatch):
    import time
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.set_last_briefing_at(user_id, time.time())  # just now
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    sent = db_briefing.sweep_daily_briefings()

    assert sent == 0
    assert temp_db.list_chat_messages(user_id) == []


def test_sweep_sends_again_after_the_interval_elapses(temp_db, monkeypatch):
    import time
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.set_last_briefing_at(user_id, time.time() - db_briefing.DAILY_BRIEFING_MIN_INTERVAL_SECONDS - 1)
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    sent = db_briefing.sweep_daily_briefings()

    assert sent == 1


def test_sweep_sends_a_push_notification(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh", "auth")
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    sent_calls = []
    monkeypatch.setattr(auth, "send_push_notification", lambda sub, title, body, status: sent_calls.append(title))
    db_briefing.sweep_daily_briefings()

    assert sent_calls == ["Your daily briefing is ready"]


def test_sweep_isolated_per_user_on_failure(temp_db, monkeypatch):
    u1 = temp_db.create_user("a@example.com", "h", "s")
    u2 = temp_db.create_user("b@example.com", "h", "s")
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    real_build = db_briefing.build_briefing

    def flaky_build(user_id):
        if user_id == u1:
            raise RuntimeError("synthetic failure")
        return real_build(user_id)

    monkeypatch.setattr(db_briefing, "build_briefing", flaky_build)

    sent = db_briefing.sweep_daily_briefings()  # must not raise

    assert sent == 1
    assert temp_db.get_user_by_id(u1)["last_briefing_at"] is None  # not advanced -- retried next sweep
    assert temp_db.get_user_by_id(u2)["last_briefing_at"] is not None


# ---------------------------------------------------------------- chat_router wiring

def test_handle_daily_briefing_advances_the_cursor(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    result = cr._handle_daily_briefing(user_id)

    assert "No holdings" in result["text"]
    assert temp_db.get_user_by_id(user_id)["last_briefing_at"] is not None


def test_handle_chat_message_routes_daily_briefing_with_its_own_ticker(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    db.set_rating_alert_seen(user_id, "TCS", "Sell")
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])  # no ticker in "give me my briefing" itself
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=20: "INTENT: daily_briefing",
    }))
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    result = cr.handle_chat_message(user_id, "give me my briefing")

    assert result["intent"] == "daily_briefing"
    assert result["ticker"] == "TCS"  # from the briefing's own content, not classify_intent's pre-pass
    assert "TCS moved to Sell" in result["reply"]


# ---------------------------------------------------------------- REST: sweep endpoint

def test_sweep_daily_briefings_endpoint_returns_a_count(client, monkeypatch):
    monkeypatch.setattr(main, "sweep_daily_briefings", lambda: 3)

    resp = client.post("/v1/internal/sweep/daily-briefings")

    assert resp.status_code == 200
    assert resp.json() == {"sent": 3}


def test_daily_briefing_intent_works_on_demand_via_chat(client, monkeypatch, auth_headers):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=20: "INTENT: daily_briefing",
    }))
    monkeypatch.setattr(db_briefing, "HostedProvider", _RaisingProvider)

    resp = client.post("/v1/chat", json={"message": "give me my briefing"}, headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json()["intent"] == "daily_briefing"
    assert "No holdings in the portfolio yet" in resp.json()["reply"]
