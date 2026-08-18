"""
Tests for simulated paper-trading orders:
- db.py's execute_order / list_orders
- app/api/main.py's POST /v1/orders, GET /v1/orders

NO real broker, NO real network trade -- every quote is monkeypatched,
same "never hit real network, patch main.get_quote directly" principle
as test_portfolio.py/test_watchlist.py.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import db, jobs
from app.api import main
from app.api.main import app
from app.core import llm_provider as lp
from app.reporting import portfolio_summary


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
    resp = client.post("/v1/auth/signup", json={"email": "orders@example.com", "password": "orderspassword"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_quote(price=100.0, change_pct=1.5, currency="USD"):
    return {"price": price, "change_pct": change_pct, "currency": currency}


# ---------------------------------------------------------------- db.execute_order

def test_execute_order_buy_opens_a_new_position(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    result = temp_db.execute_order(user_id, "AAPL", "BUY", 10, 100.0, "USD")
    assert result["new_holding_quantity"] == 10

    holdings = {h["ticker"]: h for h in temp_db.get_portfolio_holdings(user_id)}
    assert holdings["AAPL"]["quantity"] == 10
    assert holdings["AAPL"]["avg_cost"] == 100.0


def test_execute_order_buy_extends_an_existing_position_with_weighted_average(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "AAPL", "BUY", 10, 100.0, "USD")
    temp_db.execute_order(user_id, "AAPL", "BUY", 10, 200.0, "USD")

    holdings = {h["ticker"]: h for h in temp_db.get_portfolio_holdings(user_id)}
    # (10*100 + 10*200) / 20 = 150
    assert holdings["AAPL"]["quantity"] == 20
    assert holdings["AAPL"]["avg_cost"] == pytest.approx(150.0)


def test_execute_order_sell_reduces_quantity_without_changing_avg_cost(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "AAPL", "BUY", 10, 100.0, "USD")
    temp_db.execute_order(user_id, "AAPL", "SELL", 4, 150.0, "USD")

    holdings = {h["ticker"]: h for h in temp_db.get_portfolio_holdings(user_id)}
    assert holdings["AAPL"]["quantity"] == 6
    assert holdings["AAPL"]["avg_cost"] == 100.0  # unchanged by the sale


def test_execute_order_sell_that_zeroes_out_removes_the_holding(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "AAPL", "BUY", 10, 100.0, "USD")
    temp_db.execute_order(user_id, "AAPL", "SELL", 10, 150.0, "USD")

    assert temp_db.get_portfolio_holdings(user_id) == []


def test_execute_order_sell_more_than_held_raises_value_error(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "AAPL", "BUY", 5, 100.0, "USD")
    with pytest.raises(ValueError):
        temp_db.execute_order(user_id, "AAPL", "SELL", 10, 100.0, "USD")

    # The rejected sell must not have partially applied.
    holdings = {h["ticker"]: h for h in temp_db.get_portfolio_holdings(user_id)}
    assert holdings["AAPL"]["quantity"] == 5


def test_execute_order_sell_with_no_existing_holding_raises_value_error(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    with pytest.raises(ValueError):
        temp_db.execute_order(user_id, "AAPL", "SELL", 1, 100.0, "USD")


def test_execute_order_stores_a_supplied_rationale(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    result = temp_db.execute_order(user_id, "AAPL", "BUY", 10, 100.0, "USD", rationale="rating changed to Sell on 2026-08-20")

    assert result["rationale"] == "rating changed to Sell on 2026-08-20"
    orders = temp_db.list_orders(user_id)
    assert orders[0]["rationale"] == "rating changed to Sell on 2026-08-20"


def test_execute_order_defaults_rationale_to_none(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    result = temp_db.execute_order(user_id, "AAPL", "BUY", 10, 100.0, "USD")

    assert result["rationale"] is None
    assert temp_db.list_orders(user_id)[0]["rationale"] is None


def test_list_orders_returns_most_recent_first(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "AAPL", "BUY", 10, 100.0, "USD")
    temp_db.execute_order(user_id, "MSFT", "BUY", 5, 200.0, "USD")

    orders = temp_db.list_orders(user_id)
    assert [o["ticker"] for o in orders] == ["MSFT", "AAPL"]


def test_list_orders_isolated_per_user(temp_db):
    u1 = temp_db.create_user("a@example.com", "h", "s")
    u2 = temp_db.create_user("b@example.com", "h", "s")
    temp_db.execute_order(u1, "AAPL", "BUY", 10, 100.0, "USD")

    assert temp_db.list_orders(u2) == []


# ---------------------------------------------------------------- POST /v1/orders

def test_place_buy_order_returns_the_fill_and_updates_portfolio(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=150.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=150.0))

    resp = client.post("/v1/orders", json={"ticker": "aapl", "side": "BUY", "quantity": 4}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"  # normalized uppercase
    assert body["execution_price"] == 150.0
    assert body["currency"] == "USD"
    assert body["new_holding_quantity"] == 4
    # Placed directly through the REST endpoint (the UI order form), not
    # a chat proposal -- no rationale to report, and none fabricated.
    assert body["rationale"] is None

    portfolio = client.get("/v1/portfolio", headers=auth_headers).json()
    assert portfolio["holdings"][0]["ticker"] == "AAPL"
    assert portfolio["holdings"][0]["quantity"] == 4
    assert portfolio["holdings"][0]["avg_cost"] == 150.0


def test_place_sell_order_reduces_the_holding(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=100.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=100.0))
    client.post("/v1/orders", json={"ticker": "AAPL", "side": "BUY", "quantity": 10}, headers=auth_headers)

    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=120.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=120.0))
    resp = client.post("/v1/orders", json={"ticker": "AAPL", "side": "SELL", "quantity": 3}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["new_holding_quantity"] == 7

    portfolio = client.get("/v1/portfolio", headers=auth_headers).json()
    assert portfolio["holdings"][0]["quantity"] == 7
    assert portfolio["holdings"][0]["avg_cost"] == 100.0


def test_selling_more_than_held_returns_insufficient_shares_error(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/orders", json={"ticker": "AAPL", "side": "BUY", "quantity": 2}, headers=auth_headers)

    resp = client.post("/v1/orders", json={"ticker": "AAPL", "side": "SELL", "quantity": 5}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["code"] == "INSUFFICIENT_SHARES"


def test_place_order_with_an_invalid_ticker_returns_ticker_not_found(client, monkeypatch, auth_headers):
    def raise_not_found(ticker):
        from app.data.market_data import TickerNotFoundError
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(main, "get_quote", raise_not_found)
    monkeypatch.setattr(portfolio_summary, "get_quote", raise_not_found)
    monkeypatch.setattr(main, "resolve_companies", lambda q: [])

    resp = client.post("/v1/orders", json={"ticker": "ZZZZZZ", "side": "BUY", "quantity": 1}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


def test_place_order_rejects_zero_or_negative_quantity(client, auth_headers):
    resp = client.post("/v1/orders", json={"ticker": "AAPL", "side": "BUY", "quantity": 0}, headers=auth_headers)
    assert resp.status_code == 422


def test_place_order_rejects_an_invalid_side(client, auth_headers):
    resp = client.post("/v1/orders", json={"ticker": "AAPL", "side": "HOLD", "quantity": 1}, headers=auth_headers)
    assert resp.status_code == 422


def test_place_order_requires_a_session(client):
    resp = client.post("/v1/orders", json={"ticker": "AAPL", "side": "BUY", "quantity": 1})
    assert resp.status_code == 401


def test_place_order_captures_a_non_usd_currency(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=1000.0, currency="INR"))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=1000.0, currency="INR"))
    resp = client.post("/v1/orders", json={"ticker": "BAJFINANCE.NS", "side": "BUY", "quantity": 2}, headers=auth_headers)
    assert resp.json()["currency"] == "INR"


# ---------------------------------------------------------------- GET /v1/orders

def test_get_orders_returns_order_history(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=100.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=100.0))
    client.post("/v1/orders", json={"ticker": "AAPL", "side": "BUY", "quantity": 5}, headers=auth_headers)
    client.post("/v1/orders", json={"ticker": "MSFT", "side": "BUY", "quantity": 2}, headers=auth_headers)

    resp = client.get("/v1/orders", headers=auth_headers)
    assert resp.status_code == 200
    orders = resp.json()["orders"]
    assert [o["ticker"] for o in orders] == ["MSFT", "AAPL"]


def test_get_orders_requires_a_session(client):
    resp = client.get("/v1/orders")
    assert resp.status_code == 401


def test_get_orders_caps_the_limit_at_50(client, auth_headers):
    resp = client.get("/v1/orders?limit=100", headers=auth_headers)
    assert resp.status_code == 422


def test_get_orders_empty_for_a_new_user(client, auth_headers):
    resp = client.get("/v1/orders", headers=auth_headers)
    assert resp.json()["orders"] == []
