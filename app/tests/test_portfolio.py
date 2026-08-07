"""
Tests for the self-reported portfolio endpoints (app/api/main.py's
GET/POST /v1/portfolio, DELETE /v1/portfolio/{ticker}).

Same "never hit real network, patch main.get_quote directly" principle
as test_watchlist.py -- main.py does `from app.data.market_data import
get_quote`, so main.get_quote is a separate bound name.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import db, jobs
from app.api import main
from app.api.main import app
from app.core import llm_provider as lp
from app.core.research_context import ResearchContext
from app.data.market_data import TickerNotFoundError


class _StubAgent:
    """Returns a fully-populated fake ResearchContext instantly -- same
    spirit as test_watchlist.py's own stub, trimmed to what these tests
    need (a ticker + rating to exercise the Portfolio's research pill)."""

    def run(self, question):
        context = ResearchContext(ticker="AAPL", question=question)
        context.report_data = {"ticker": "AAPL", "recommendation": {"rating": "Buy"}}
        context.normalized_financials = None
        context.pdf_bytes = b"%PDF-1.4 fake pdf"
        return context


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setattr(lp, "_provider", None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(jobs, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(jobs, "ORCHESTRATORS", {**jobs.ORCHESTRATORS, "hand_rolled": _StubAgent})
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    resp = client.post("/v1/auth/signup", json={"email": "portfolio@example.com", "password": "portfoliopassword"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_quote(price=200.0, change_pct=1.5, previous_close=None):
    return {"price": price, "change_pct": change_pct, "previous_close": previous_close}


# ---------------------------------------------------------------- add/list/remove

def test_add_list_remove_round_trip(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=200.0))

    resp = client.post("/v1/portfolio", json={"ticker": "aapl", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get("/v1/portfolio", headers=auth_headers)
    assert resp.status_code == 200
    holdings = resp.json()["holdings"]
    assert len(holdings) == 1
    h = holdings[0]
    assert h["ticker"] == "AAPL"
    assert h["quantity"] == 10
    assert h["avg_cost"] == 150.0
    assert h["price"] == 200.0
    assert h["cost_basis"] == 1500.0
    assert h["market_value"] == 2000.0
    assert h["unrealized_pnl"] == 500.0
    assert h["unrealized_pnl_pct"] == pytest.approx(33.333, rel=1e-3)

    summary = resp.json()["summary"]
    assert summary["total_market_value"] == 2000.0
    assert summary["total_cost_basis"] == 1500.0
    assert summary["total_unrealized_pnl"] == 500.0

    resp = client.delete("/v1/portfolio/AAPL", headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get("/v1/portfolio", headers=auth_headers)
    assert resp.json()["holdings"] == []
    assert resp.json()["summary"]["total_market_value"] is None


def test_re_adding_the_same_ticker_replaces_quantity_and_avg_cost(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 5, "avg_cost": 180.0}, headers=auth_headers)

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert len(holdings) == 1
    assert holdings[0]["quantity"] == 5
    assert holdings[0]["avg_cost"] == 180.0


def test_removing_a_holding_not_in_the_portfolio_is_not_an_error(client, auth_headers):
    resp = client.delete("/v1/portfolio/NEVERADDED", headers=auth_headers)
    assert resp.status_code == 200


def test_portfolio_requires_a_session(client):
    resp = client.get("/v1/portfolio")
    assert resp.status_code == 401


# ---------------------------------------------------------------- validation

def test_adding_an_invalid_ticker_returns_ticker_not_found(client, monkeypatch, auth_headers):
    def raise_not_found(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(main, "get_quote", raise_not_found)

    resp = client.post("/v1/portfolio", json={"ticker": "ZZZZZZ", "quantity": 1, "avg_cost": 10.0}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


@pytest.mark.parametrize("quantity,avg_cost", [(0, 10.0), (-5, 10.0), (10, 0), (10, -1.0)])
def test_non_positive_quantity_or_cost_is_rejected(client, monkeypatch, auth_headers, quantity, avg_cost):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    resp = client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": quantity, "avg_cost": avg_cost}, headers=auth_headers)
    assert resp.status_code == 422


def test_adding_a_fuzzy_company_name_falls_back_to_resolve_companies(client, monkeypatch, auth_headers):
    def flaky_quote(ticker):
        if ticker == "BAJAJ FINANCE":
            raise TickerNotFoundError(ticker)
        assert ticker == "BAJFINANCE.NS"
        return _fake_quote()

    monkeypatch.setattr(main, "get_quote", flaky_quote)
    monkeypatch.setattr(main, "resolve_companies", lambda q: ["BAJFINANCE.NS"])

    resp = client.post("/v1/portfolio", json={"ticker": "Bajaj Finance", "quantity": 1, "avg_cost": 10.0}, headers=auth_headers)
    assert resp.status_code == 200

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert [h["ticker"] for h in holdings] == ["BAJFINANCE.NS"]


# ---------------------------------------------------------------- per-ticker isolation

def test_one_bad_ticker_does_not_500_the_whole_portfolio(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "MSFT", "quantity": 5, "avg_cost": 100.0}, headers=auth_headers)

    def flaky_quote(ticker):
        if ticker == "MSFT":
            raise TickerNotFoundError(ticker)
        return _fake_quote()

    monkeypatch.setattr(main, "get_quote", flaky_quote)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    assert resp.status_code == 200
    holdings = {h["ticker"]: h for h in resp.json()["holdings"]}
    assert holdings["AAPL"]["price"] == 200.0
    assert holdings["MSFT"]["price"] is None
    assert holdings["MSFT"]["market_value"] is None
    assert holdings["MSFT"]["unrealized_pnl"] is None
    # Cost basis is knowable even without a live quote -- only
    # market-value-dependent fields should be null.
    assert holdings["MSFT"]["cost_basis"] == 500.0

    # Portfolio-level total_market_value must reflect only the
    # tickers with a live quote, not silently include None.
    summary = resp.json()["summary"]
    assert summary["total_market_value"] == 2000.0


# ---------------------------------------------------------------- buy_date

def test_buy_date_is_persisted_and_returned(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    resp = client.post(
        "/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "buy_date": "2026-01-15"},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["buy_date"] == "2026-01-15"


def test_buy_date_is_optional(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    resp = client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)
    assert resp.status_code == 200

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["buy_date"] is None


def test_buy_date_rejects_a_malformed_date(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    resp = client.post(
        "/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "buy_date": "not-a-date"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_re_adding_a_holding_updates_its_buy_date(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "buy_date": "2026-01-15"}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 5, "avg_cost": 180.0, "buy_date": "2026-02-01"}, headers=auth_headers)

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["buy_date"] == "2026-02-01"


# ---------------------------------------------------------------- research pill (live rating)

def test_holding_shows_no_rating_when_never_researched(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["rating"] is None


def test_holding_rating_reflects_the_users_last_completed_report(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)

    submit_resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    job_id = submit_resp.json()["job_id"]

    import time
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.get(f"/v1/research/{job_id}", headers=auth_headers).json()["status"] == db.STATUS_DONE:
            break
        time.sleep(0.05)

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["rating"] == "Buy"


# ---------------------------------------------------------------- today's P&L

def test_today_pnl_is_computed_from_previous_close(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=110.0, previous_close=100.0))
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    holding = resp.json()["holdings"][0]
    assert holding["today_pnl"] == pytest.approx(100.0)  # 10 * (110 - 100)

    summary = resp.json()["summary"]
    assert summary["total_today_pnl"] == pytest.approx(100.0)
    # Yesterday's value = 1100 (today's market value) - 100 (today's gain) = 1000
    assert summary["total_today_pnl_pct"] == pytest.approx(10.0)


def test_today_pnl_is_none_without_a_previous_close(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=110.0, previous_close=None))
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    assert resp.json()["holdings"][0]["today_pnl"] is None
    assert resp.json()["summary"]["total_today_pnl"] is None
    assert resp.json()["summary"]["total_today_pnl_pct"] is None


def test_today_pnl_ignores_holdings_with_a_failed_quote(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "MSFT", "quantity": 5, "avg_cost": 100.0}, headers=auth_headers)

    def flaky_quote(ticker):
        if ticker == "MSFT":
            raise TickerNotFoundError(ticker)
        return _fake_quote(price=110.0, previous_close=100.0)

    monkeypatch.setattr(main, "get_quote", flaky_quote)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    holdings = {h["ticker"]: h for h in resp.json()["holdings"]}
    assert holdings["AAPL"]["today_pnl"] == pytest.approx(100.0)
    assert holdings["MSFT"]["today_pnl"] is None
    # Only AAPL's gain counted -- MSFT's missing quote must not corrupt the total.
    assert resp.json()["summary"]["total_today_pnl"] == pytest.approx(100.0)
