"""
Tests for the self-reported portfolio endpoints (app/api/main.py's
GET/POST /v1/portfolio, DELETE /v1/portfolio/{ticker}).

Same "never hit real network, patch main.get_quote directly" principle
as test_watchlist.py -- main.py does `from app.data.market_data import
get_quote`, so main.get_quote is a separate bound name. GET /v1/portfolio
specifically now delegates to app/reporting/portfolio_summary.py's
build_portfolio_view() (extracted so the fast-chat "portfolio status"
intent can reuse it), which has its OWN separately-bound get_quote/
get_usd_conversion_rate -- every test that reads the GET response
patches both main's and portfolio_summary's names.
"""

import pytest
from fastapi.testclient import TestClient

from app.api import db, jobs
from app.api import main
from app.api.main import app
from app.core import llm_provider as lp
from app.core.research_context import ResearchContext
from app.data.market_data import TickerNotFoundError
from app.reporting import portfolio_summary


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


def _fake_quote(price=200.0, change_pct=1.5, previous_close=None, currency="USD"):
    return {"price": price, "change_pct": change_pct, "previous_close": previous_close, "currency": currency}


# ---------------------------------------------------------------- add/list/remove

def test_add_list_remove_round_trip(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=200.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=200.0))

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
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
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
    monkeypatch.setattr(portfolio_summary, "get_quote", raise_not_found)

    resp = client.post("/v1/portfolio", json={"ticker": "ZZZZZZ", "quantity": 1, "avg_cost": 10.0}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


@pytest.mark.parametrize("quantity,avg_cost", [(0, 10.0), (-5, 10.0), (10, 0), (10, -1.0)])
def test_non_positive_quantity_or_cost_is_rejected(client, monkeypatch, auth_headers, quantity, avg_cost):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    resp = client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": quantity, "avg_cost": avg_cost}, headers=auth_headers)
    assert resp.status_code == 422


def test_adding_a_fuzzy_company_name_falls_back_to_resolve_companies(client, monkeypatch, auth_headers):
    def flaky_quote(ticker):
        if ticker == "BAJAJ FINANCE":
            raise TickerNotFoundError(ticker)
        assert ticker == "BAJFINANCE.NS"
        return _fake_quote()

    monkeypatch.setattr(main, "get_quote", flaky_quote)
    monkeypatch.setattr(portfolio_summary, "get_quote", flaky_quote)
    monkeypatch.setattr(main, "resolve_companies", lambda q: ["BAJFINANCE.NS"])

    resp = client.post("/v1/portfolio", json={"ticker": "Bajaj Finance", "quantity": 1, "avg_cost": 10.0}, headers=auth_headers)
    assert resp.status_code == 200

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert [h["ticker"] for h in holdings] == ["BAJFINANCE.NS"]


# ---------------------------------------------------------------- per-ticker isolation

def test_one_bad_ticker_does_not_500_the_whole_portfolio(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "MSFT", "quantity": 5, "avg_cost": 100.0}, headers=auth_headers)

    def flaky_quote(ticker):
        if ticker == "MSFT":
            raise TickerNotFoundError(ticker)
        return _fake_quote()

    monkeypatch.setattr(main, "get_quote", flaky_quote)
    monkeypatch.setattr(portfolio_summary, "get_quote", flaky_quote)

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
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    resp = client.post(
        "/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "buy_date": "2026-01-15"},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["buy_date"] == "2026-01-15"


def test_buy_date_is_optional(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    resp = client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)
    assert resp.status_code == 200

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["buy_date"] is None


def test_buy_date_rejects_a_malformed_date(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    resp = client.post(
        "/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "buy_date": "not-a-date"},
        headers=auth_headers,
    )
    assert resp.status_code == 422


def test_re_adding_a_holding_updates_its_buy_date(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0, "buy_date": "2026-01-15"}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 5, "avg_cost": 180.0, "buy_date": "2026-02-01"}, headers=auth_headers)

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["buy_date"] == "2026-02-01"


# ---------------------------------------------------------------- research pill (live rating)

def test_holding_shows_no_rating_when_never_researched(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["rating"] is None


def test_holding_rating_reflects_the_users_last_completed_report(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 150.0}, headers=auth_headers)

    submit_resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    job_id = submit_resp.json()["job_id"]

    import time
    # 15s, not 5s -- GitHub Actions' shared 2-core runners are
    # consistently slower than a dev machine, and this test was caught
    # failing there (CI runs 2026-08-05 through 2026-08-07, every single
    # one) while passing reliably in isolation locally. 5s was already
    # observed flaking locally too under heavy concurrent CPU load
    # earlier this session -- CI just hits that same margin every time,
    # not occasionally.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if client.get(f"/v1/research/{job_id}", headers=auth_headers).json()["status"] == db.STATUS_DONE:
            break
        time.sleep(0.05)

    holdings = client.get("/v1/portfolio", headers=auth_headers).json()["holdings"]
    assert holdings[0]["rating"] == "Buy"


# ---------------------------------------------------------------- today's P&L

def test_today_pnl_is_computed_from_previous_close(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=110.0, previous_close=100.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=110.0, previous_close=100.0))
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
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=110.0, previous_close=None))
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    assert resp.json()["holdings"][0]["today_pnl"] is None
    assert resp.json()["summary"]["total_today_pnl"] is None
    assert resp.json()["summary"]["total_today_pnl_pct"] is None


def test_today_pnl_ignores_holdings_with_a_failed_quote(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 100.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "MSFT", "quantity": 5, "avg_cost": 100.0}, headers=auth_headers)

    def flaky_quote(ticker):
        if ticker == "MSFT":
            raise TickerNotFoundError(ticker)
        return _fake_quote(price=110.0, previous_close=100.0)

    monkeypatch.setattr(main, "get_quote", flaky_quote)
    monkeypatch.setattr(portfolio_summary, "get_quote", flaky_quote)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    holdings = {h["ticker"]: h for h in resp.json()["holdings"]}
    assert holdings["AAPL"]["today_pnl"] == pytest.approx(100.0)
    assert holdings["MSFT"]["today_pnl"] is None
    # Only AAPL's gain counted -- MSFT's missing quote must not corrupt the total.
    assert resp.json()["summary"]["total_today_pnl"] == pytest.approx(100.0)


# ---------------------------------------------------------------- currency

def test_holding_shows_its_own_native_currency_not_converted(client, monkeypatch, auth_headers):
    # A rupee-priced holding must never be silently relabeled as
    # dollars -- currency comes straight from the quote, and the
    # per-holding market_value/cost_basis stay in that native currency
    # (only the aggregate summary below gets converted).
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=1000.0, currency="INR"))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=1000.0, currency="INR"))
    monkeypatch.setattr(main, "get_usd_conversion_rate", lambda currency: 1.0 / 95.0 if currency == "INR" else 1.0)
    monkeypatch.setattr(portfolio_summary, "get_usd_conversion_rate", lambda currency: 1.0 / 95.0 if currency == "INR" else 1.0)
    client.post("/v1/portfolio", json={"ticker": "BAJFINANCE.NS", "quantity": 2, "avg_cost": 900.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    holding = resp.json()["holdings"][0]
    assert holding["currency"] == "INR"
    assert holding["market_value"] == pytest.approx(2000.0)
    assert holding["cost_basis"] == pytest.approx(1800.0)


def test_summary_converts_mixed_currency_holdings_to_usd_equivalent(client, monkeypatch, auth_headers):
    def fake_quote(ticker):
        if ticker == "AAPL":
            return _fake_quote(price=100.0, currency="USD")
        return _fake_quote(price=1000.0, currency="INR")

    monkeypatch.setattr(main, "get_quote", fake_quote)
    monkeypatch.setattr(portfolio_summary, "get_quote", fake_quote)
    monkeypatch.setattr(main, "get_usd_conversion_rate", lambda currency: 1.0 if currency == "USD" else 0.01)
    monkeypatch.setattr(portfolio_summary, "get_usd_conversion_rate", lambda currency: 1.0 if currency == "USD" else 0.01)
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "BAJFINANCE.NS", "quantity": 5, "avg_cost": 900.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    summary = resp.json()["summary"]
    # AAPL: 10*100=1000 USD market value, 10*90=900 USD cost basis (rate 1.0).
    # BAJFINANCE.NS: 5*1000=5000 INR market value * 0.01 = 50 USD;
    # 5*900=4500 INR cost basis * 0.01 = 45 USD.
    assert summary["total_market_value"] == pytest.approx(1050.0)
    assert summary["total_cost_basis"] == pytest.approx(945.0)
    assert summary["currency"] == "USD"
    assert summary["mixed_currency"] is True


def test_summary_mixed_currency_is_false_for_an_all_usd_portfolio(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    assert resp.json()["summary"]["mixed_currency"] is False


def test_summary_excludes_a_holding_with_no_known_fx_rate(client, monkeypatch, auth_headers):
    # A currency this app has no FX ticker for (anything besides USD/
    # INR) must not silently get treated as 1:1 with USD -- the holding
    # still displays correctly on its own row, but its value is left
    # out of the USD aggregate rather than corrupting the total.
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=100.0, currency="EUR"))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=100.0, currency="EUR"))
    monkeypatch.setattr(main, "get_usd_conversion_rate", lambda currency: None)
    monkeypatch.setattr(portfolio_summary, "get_usd_conversion_rate", lambda currency: None)
    client.post("/v1/portfolio", json={"ticker": "SAP.DE", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio", headers=auth_headers)
    body = resp.json()
    assert body["holdings"][0]["market_value"] == pytest.approx(1000.0)
    assert body["summary"]["excluded_from_summary"] is True


# ---------------------------------------------------------------- combined portfolio analysis

def _complete_job_with_rating(user_id, ticker, rating):
    job_id = db.create_job("q", "hand_rolled", user_id=user_id)
    db.mark_running(job_id)
    result = {"ticker": ticker, "report_data": {"recommendation": {"rating": rating}}}
    db.mark_done(job_id, result, pdf_path="/tmp/fake.pdf")


def test_portfolio_analysis_counts_ratings_across_holdings(client, monkeypatch, auth_headers):
    user_id = client.get("/v1/auth/me", headers=auth_headers).json()["user_id"]
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=100.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=100.0))
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "MSFT", "quantity": 5, "avg_cost": 90.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "TSLA", "quantity": 1, "avg_cost": 90.0}, headers=auth_headers)

    _complete_job_with_rating(user_id, "AAPL", "Buy")
    _complete_job_with_rating(user_id, "MSFT", "Buy")
    _complete_job_with_rating(user_id, "TSLA", "Sell")

    resp = client.get("/v1/portfolio/analysis", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["holdings_total"] == 3
    assert body["holdings_researched"] == 3
    assert body["rating_counts"] == {"Buy": 2, "Hold": 0, "Sell": 1}
    assert body["unresearched_tickers"] == []


def test_portfolio_analysis_tracks_unresearched_holdings_separately(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote(price=100.0))
    monkeypatch.setattr(portfolio_summary, "get_quote", lambda ticker: _fake_quote(price=100.0))
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)

    resp = client.get("/v1/portfolio/analysis", headers=auth_headers)
    body = resp.json()
    assert body["holdings_researched"] == 0
    assert body["unresearched_tickers"] == ["AAPL"]
    assert body["rating_counts"] == {"Buy": 0, "Hold": 0, "Sell": 0}


def test_portfolio_analysis_value_weights_by_usd_equivalent_market_value(client, monkeypatch, auth_headers):
    user_id = client.get("/v1/auth/me", headers=auth_headers).json()["user_id"]

    def fake_quote(ticker):
        return _fake_quote(price=100.0 if ticker == "AAPL" else 10.0)

    monkeypatch.setattr(main, "get_quote", fake_quote)
    monkeypatch.setattr(portfolio_summary, "get_quote", fake_quote)
    # AAPL: 10 * 100 = 1000 (Buy). MSFT: 100 * 10 = 1000 (Sell). Equal value -> 50/50.
    client.post("/v1/portfolio", json={"ticker": "AAPL", "quantity": 10, "avg_cost": 90.0}, headers=auth_headers)
    client.post("/v1/portfolio", json={"ticker": "MSFT", "quantity": 100, "avg_cost": 9.0}, headers=auth_headers)

    _complete_job_with_rating(user_id, "AAPL", "Buy")
    _complete_job_with_rating(user_id, "MSFT", "Sell")

    resp = client.get("/v1/portfolio/analysis", headers=auth_headers)
    weighted = resp.json()["value_weighted_pct"]
    assert weighted["Buy"] == pytest.approx(50.0)
    assert weighted["Sell"] == pytest.approx(50.0)


def test_portfolio_analysis_value_weighted_pct_is_none_with_no_researched_value(client, auth_headers):
    resp = client.get("/v1/portfolio/analysis", headers=auth_headers)
    assert resp.json()["value_weighted_pct"] is None


def test_portfolio_analysis_requires_a_session(client):
    resp = client.get("/v1/portfolio/analysis")
    assert resp.status_code == 401
