"""
Tests for Feature 3 (price-triggered alerts):
- db.py's price_alerts CRUD (create/list/get_active/mark_triggered)
- app/reasoning/price_alerts.py's sweep_price_alerts

Real sqlite temp DB throughout (same pattern as test_rating_alerts.py);
get_quote and auth.send_push_notification are the only things
monkeypatched -- never a real network/push call.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import auth, db, jobs, main
from app.api.main import app
from app.core import llm_provider as lp
from app.reasoning import price_alerts as pa


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Same client/auth_headers fixture pair as test_orders.py.
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setattr(lp, "_provider", None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(jobs, "REPORTS_DIR", tmp_path / "reports")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    resp = client.post("/v1/auth/signup", json={"email": "alerts@example.com", "password": "alertspassword"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_quote(price=100.0, change_pct=1.5, currency="USD"):
    return {"price": price, "change_pct": change_pct, "currency": currency}


@pytest.fixture
def sent_notifications(monkeypatch):
    calls = []

    def fake_send(subscription, title, body, status):
        calls.append((subscription["endpoint"], title, body, status))

    monkeypatch.setattr(auth, "send_push_notification", fake_send)
    return calls


# ---------------------------------------------------------------- db.py CRUD

def test_create_price_alert_defaults_auto_execute_to_false(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)

    alerts = temp_db.list_price_alerts(user_id)
    assert alerts[0]["auto_execute"] == 0


def test_create_price_alert_rejects_an_invalid_side(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    with pytest.raises(ValueError):
        temp_db.create_price_alert(user_id, "AAPL", "HOLD", "below", 180.0)


def test_create_price_alert_rejects_an_invalid_direction(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    with pytest.raises(ValueError):
        temp_db.create_price_alert(user_id, "AAPL", "SELL", "sideways", 180.0)


def test_list_price_alerts_excludes_triggered_by_default(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    alert_id = temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)
    temp_db.mark_price_alert_triggered(alert_id)

    assert temp_db.list_price_alerts(user_id) == []
    assert len(temp_db.list_price_alerts(user_id, active_only=False)) == 1


def test_list_price_alerts_isolated_per_user(temp_db):
    u1 = temp_db.create_user("a@example.com", "h", "s")
    u2 = temp_db.create_user("b@example.com", "h", "s")
    temp_db.create_price_alert(u1, "AAPL", "SELL", "below", 180.0)

    assert temp_db.list_price_alerts(u2) == []


def test_get_active_price_alerts_spans_all_users(temp_db):
    u1 = temp_db.create_user("a@example.com", "h", "s")
    u2 = temp_db.create_user("b@example.com", "h", "s")
    temp_db.create_price_alert(u1, "AAPL", "SELL", "below", 180.0)
    temp_db.create_price_alert(u2, "TCS", "BUY", "below", 4000.0)

    tickers = {a["ticker"] for a in temp_db.get_active_price_alerts()}
    assert tickers == {"AAPL", "TCS"}


def test_mark_price_alert_triggered_is_one_shot(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    alert_id = temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)

    temp_db.mark_price_alert_triggered(alert_id)
    first_triggered_at = temp_db.list_price_alerts(user_id, active_only=False)[0]["triggered_at"]

    temp_db.mark_price_alert_triggered(alert_id)  # a hypothetical duplicate/late sweep call
    second_triggered_at = temp_db.list_price_alerts(user_id, active_only=False)[0]["triggered_at"]

    assert first_triggered_at == second_triggered_at  # not re-stamped


# ---------------------------------------------------------------- sweep_price_alerts

def test_sweep_does_not_fire_an_alert_whose_condition_is_not_yet_met(temp_db, monkeypatch, sent_notifications):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 100.0)  # fires at <=$100
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})  # well above

    fired = pa.sweep_price_alerts()

    assert fired == 0
    assert sent_notifications == []
    assert temp_db.list_price_alerts(user_id)[0]["triggered_at"] is None


def test_sweep_fires_a_below_alert_once_price_has_dropped_to_or_under_target(temp_db, monkeypatch, sent_notifications):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh", "auth")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 175.0, "currency": "USD"})

    fired = pa.sweep_price_alerts()

    assert fired == 1
    assert len(sent_notifications) == 1
    assert "dropped below" in sent_notifications[0][2]
    assert temp_db.list_price_alerts(user_id, active_only=False)[0]["triggered_at"] is not None


def test_sweep_fires_an_above_alert_once_price_has_risen_to_or_over_target(temp_db, monkeypatch, sent_notifications):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh", "auth")
    temp_db.create_price_alert(user_id, "AAPL", "BUY", "above", 200.0)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 205.0, "currency": "USD"})

    fired = pa.sweep_price_alerts()

    assert fired == 1
    assert "rose above" in sent_notifications[0][2]


def test_sweep_fires_exactly_once_and_never_again_after_triggering(temp_db, monkeypatch, sent_notifications):
    # The spec's own acceptance check.
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh", "auth")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})  # already past

    first = pa.sweep_price_alerts()
    second = pa.sweep_price_alerts()

    assert first == 1
    assert second == 0
    assert len(sent_notifications) == 1
    assert temp_db.list_price_alerts(user_id) == []  # no longer active


def test_sweep_writes_a_proposal_chat_message_for_a_non_auto_alert(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})

    pa.sweep_price_alerts()

    messages = temp_db.list_chat_messages(user_id)
    assert len(messages) == 1
    assert messages[0]["ticker"] == "AAPL"
    assert "exit your position" in messages[0]["content"]


def test_sweep_survives_a_quote_failure_for_one_alert_and_still_checks_the_rest(temp_db, monkeypatch, sent_notifications):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh", "auth")
    temp_db.create_price_alert(user_id, "ZZZZ", "SELL", "below", 180.0)
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)

    def fake_quote(ticker):
        if ticker == "ZZZZ":
            raise Exception("delisted")
        return {"price": 150.0, "currency": "USD"}

    monkeypatch.setattr(pa, "get_quote", fake_quote)

    fired = pa.sweep_price_alerts()  # must not raise

    assert fired == 1
    remaining = {a["ticker"] for a in temp_db.list_price_alerts(user_id)}
    assert remaining == {"ZZZZ"}  # untouched, retried next sweep


# ---------------------------------------------------------------- auto-execute

def test_sweep_auto_execute_sell_closes_out_the_full_holding(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "AAPL", "BUY", 10, 200.0, "USD")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0, auto_execute=True)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 175.0, "currency": "USD"})

    fired = pa.sweep_price_alerts()

    assert fired == 1
    orders = temp_db.list_orders(user_id)
    assert orders[0]["side"] == "SELL"
    assert orders[0]["quantity"] == 10.0
    assert "price alert" in orders[0]["rationale"]
    assert temp_db.get_portfolio_holdings(user_id) == []


def test_sweep_auto_execute_sell_with_nothing_held_notifies_instead_of_crashing(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0, auto_execute=True)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 175.0, "currency": "USD"})

    fired = pa.sweep_price_alerts()  # must not raise

    assert fired == 1
    assert temp_db.list_orders(user_id) == []
    messages = temp_db.list_chat_messages(user_id)
    assert "nothing held to sell" in messages[0]["content"]


def test_sweep_auto_execute_buy_sizes_via_suggest_quantity(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "MSFT", "BUY", 100, 100.0, "USD")  # $10,000 portfolio value
    temp_db.set_risk_tolerance(user_id, "Aggressive")  # 10% sizing
    temp_db.create_price_alert(user_id, "AAPL", "BUY", "below", 180.0, auto_execute=True)

    def fake_quote(ticker):
        price = 100.0 if ticker == "MSFT" else 175.0
        return {"price": price, "change_pct": 0.0, "previous_close": price, "currency": "USD"}

    monkeypatch.setattr(pa, "get_quote", fake_quote)
    monkeypatch.setattr("app.reporting.portfolio_summary.get_quote", fake_quote)

    fired = pa.sweep_price_alerts()

    assert fired == 1
    orders = [o for o in temp_db.list_orders(user_id) if o["ticker"] == "AAPL"]
    assert len(orders) == 1
    assert orders[0]["side"] == "BUY"
    assert orders[0]["quantity"] == pytest.approx(5.7143, abs=0.001)  # 10% of $10,000 / $175


def test_sweep_auto_execute_notifies_with_the_result(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh", "auth")
    temp_db.execute_order(user_id, "AAPL", "BUY", 5, 200.0, "USD")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0, auto_execute=True)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 175.0, "currency": "USD"})

    sent = []
    monkeypatch.setattr(auth, "send_push_notification", lambda sub, title, body, status: sent.append(body))
    pa.sweep_price_alerts()

    assert "automatically SELL 5 shares" in sent[0]


def test_sweep_does_not_mark_triggered_when_notification_write_fails(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)
    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})
    monkeypatch.setattr(db, "add_chat_message", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk full")))

    fired = pa.sweep_price_alerts()  # must not raise

    assert fired == 0
    assert len(temp_db.list_price_alerts(user_id)) == 1  # still active, will retry


# ---------------------------------------------------------------- REST: POST/GET /v1/alerts, sweep endpoint

def test_create_and_list_alert_via_rest(client, monkeypatch, auth_headers):
    # The spec's own acceptance check, first half: a non-auto alert
    # with a target above the current price (won't fire yet) appears
    # in the list endpoint.
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=150.0))

    resp = client.post(
        "/v1/alerts",
        json={"ticker": "aapl", "side": "SELL", "direction": "above", "target_price": 200.0},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert "alert_id" in resp.json()

    listed = client.get("/v1/alerts", headers=auth_headers)
    assert listed.status_code == 200
    alerts = listed.json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["ticker"] == "AAPL"  # normalized uppercase
    assert alerts[0]["auto_execute"] == 0
    assert alerts[0]["triggered_at"] is None


def test_create_alert_rejects_an_invalid_ticker(client, monkeypatch, auth_headers):
    def raise_not_found(ticker):
        from app.data.market_data import TickerNotFoundError
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(main, "get_quote", raise_not_found)
    monkeypatch.setattr(main, "resolve_companies", lambda q: [])

    resp = client.post(
        "/v1/alerts",
        json={"ticker": "ZZZZZZ", "side": "SELL", "direction": "below", "target_price": 100.0},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


def test_create_alert_rejects_zero_or_negative_target_price(client, auth_headers):
    resp = client.post(
        "/v1/alerts",
        json={"ticker": "AAPL", "side": "SELL", "direction": "below", "target_price": 0},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_create_alert_requires_a_session(client):
    resp = client.post(
        "/v1/alerts", json={"ticker": "AAPL", "side": "SELL", "direction": "below", "target_price": 100.0},
    )
    assert resp.status_code == 401


def test_list_alerts_requires_a_session(client):
    assert client.get("/v1/alerts").status_code == 401


def test_sweep_price_alerts_endpoint_fires_the_already_past_alert(client, monkeypatch, auth_headers):
    # The spec's own acceptance check, second half: a target that's
    # already past fires exactly once on the next sweep, and the alert
    # is marked triggered so it never fires again.
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=150.0))
    client.post(
        "/v1/alerts",
        json={"ticker": "AAPL", "side": "SELL", "direction": "below", "target_price": 180.0},  # already past
        headers=auth_headers,
    )

    monkeypatch.setattr(pa, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})

    first = client.post("/v1/internal/sweep/price-alerts")
    assert first.status_code == 200
    assert first.json() == {"fired": 1}

    second = client.post("/v1/internal/sweep/price-alerts")
    assert second.json() == {"fired": 0}  # doesn't fire again

    assert client.get("/v1/alerts", headers=auth_headers).json()["alerts"] == []  # no longer active
