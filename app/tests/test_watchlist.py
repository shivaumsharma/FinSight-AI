"""
Tests for the watchlist endpoints (app/api/main.py's GET/POST
/v1/watchlist, DELETE /v1/watchlist/{ticker}).

main.py does `from app.data.market_data import get_quote`, so
main.get_quote is a separate name bound to the same function object --
monkeypatching market_data.get_quote would NOT affect what main.py
calls. Every test here patches main.get_quote directly, never real
yfinance (same "never hit real network in tests" principle as
test_market_data.py's monkeypatched stock.calendar).
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
    spirit as test_api.py's _StubAgent, trimmed to what these tests
    need (a ticker + rating to exercise get_latest_rating_for_ticker)."""

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
    # Sane default so every GET /v1/watchlist call in this file doesn't
    # need its own explicit patch just to avoid hitting real yfinance --
    # tests that care about corporate-actions behavior specifically
    # override this with their own monkeypatch.setattr call.
    monkeypatch.setattr(main, "get_corporate_actions", lambda ticker: _fake_corporate_actions())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(client):
    resp = client.post("/v1/auth/signup", json={"email": "watch@example.com", "password": "watchlistpassword"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["session_token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_quote(price=200.0, change_pct=1.5):
    return {"price": price, "change_pct": change_pct}


def _fake_corporate_actions(**overrides):
    base = {
        "next_earnings_date": "2026-10-30",
        "next_ex_dividend_date": "2026-08-10",
        "last_dividend_amount": 0.27,
        "last_split": {"date": "2020-08-31", "ratio": 4.0},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- add/list/remove

def test_add_list_remove_round_trip(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())

    resp = client.post("/v1/watchlist", json={"ticker": "aapl"}, headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get("/v1/watchlist", headers=auth_headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    # Stored uppercased regardless of the casing submitted.
    assert items[0]["ticker"] == "AAPL"
    assert items[0]["price"] == 200.0
    assert items[0]["change_pct"] == 1.5
    # No research ever run for this user -- no last-known rating yet.
    assert items[0]["rating"] is None

    resp = client.delete("/v1/watchlist/AAPL", headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get("/v1/watchlist", headers=auth_headers)
    assert resp.json()["items"] == []


def test_adding_the_same_ticker_twice_is_idempotent(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/watchlist", json={"ticker": "AAPL"}, headers=auth_headers)
    client.post("/v1/watchlist", json={"ticker": "aapl"}, headers=auth_headers)

    items = client.get("/v1/watchlist", headers=auth_headers).json()["items"]
    assert len(items) == 1


def test_removing_a_ticker_not_on_the_watchlist_is_not_an_error(client, auth_headers):
    resp = client.delete("/v1/watchlist/NEVERADDED", headers=auth_headers)
    assert resp.status_code == 200


def test_watchlist_requires_a_session(client):
    resp = client.get("/v1/watchlist")
    assert resp.status_code == 401


# ---------------------------------------------------------------- validation

def test_adding_an_invalid_ticker_returns_ticker_not_found(client, monkeypatch, auth_headers):
    def raise_not_found(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(main, "get_quote", raise_not_found)

    resp = client.post("/v1/watchlist", json={"ticker": "ZZZZZZ"}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"

    # And it must not have been added despite the failed validation.
    assert client.get("/v1/watchlist", headers=auth_headers).json()["items"] == []


def test_adding_a_fuzzy_company_name_falls_back_to_resolve_companies(client, monkeypatch, auth_headers):
    # A raw ticker lookup on "Bajaj Finance" (not a ticker at all) must
    # fail, then fall back to resolve_companies (the same NLP/NSE-aware
    # resolver the main search bar uses) to turn it into a real ticker
    # -- matches resolve_companies' own docstring example. WatchlistRequest's
    # own validator uppercases the raw input before this ever runs (see
    # main.py's WatchlistRequest._normalize_ticker), so the value this
    # handler actually sees is "BAJAJ FINANCE", not the original casing
    # -- confirmed separately that resolve_companies matches case-
    # insensitively, so this doesn't break the fallback.
    def flaky_quote(ticker):
        if ticker == "BAJAJ FINANCE":
            raise TickerNotFoundError(ticker)
        assert ticker == "BAJFINANCE.NS"
        return _fake_quote()

    monkeypatch.setattr(main, "get_quote", flaky_quote)
    monkeypatch.setattr(main, "resolve_companies", lambda q: ["BAJFINANCE.NS"])

    resp = client.post("/v1/watchlist", json={"ticker": "Bajaj Finance"}, headers=auth_headers)
    assert resp.status_code == 200

    items = client.get("/v1/watchlist", headers=auth_headers).json()["items"]
    # Stored under the RESOLVED ticker, not the raw fuzzy input.
    assert [i["ticker"] for i in items] == ["BAJFINANCE.NS"]


def test_fuzzy_name_that_resolves_to_nothing_still_404s_cleanly(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: (_ for _ in ()).throw(TickerNotFoundError(ticker)))
    monkeypatch.setattr(main, "resolve_companies", lambda q: [])

    resp = client.post("/v1/watchlist", json={"ticker": "not a real company"}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


# ---------------------------------------------------------------- per-ticker isolation

def test_one_bad_ticker_does_not_500_the_whole_watchlist(client, monkeypatch, auth_headers):
    # Add two tickers while get_quote is healthy, then make it start
    # failing for one of them -- GET /v1/watchlist must still return
    # both rows, with a null quote only for the failing one.
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/watchlist", json={"ticker": "AAPL"}, headers=auth_headers)
    client.post("/v1/watchlist", json={"ticker": "MSFT"}, headers=auth_headers)

    def flaky_quote(ticker):
        if ticker == "MSFT":
            raise TickerNotFoundError(ticker)
        return _fake_quote()

    monkeypatch.setattr(main, "get_quote", flaky_quote)

    resp = client.get("/v1/watchlist", headers=auth_headers)
    assert resp.status_code == 200
    items = {item["ticker"]: item for item in resp.json()["items"]}
    assert len(items) == 2
    assert items["AAPL"]["price"] == 200.0
    assert items["MSFT"]["price"] is None
    assert items["MSFT"]["change_pct"] is None


# ---------------------------------------------------------------- corporate actions

def test_watchlist_includes_corporate_actions(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    monkeypatch.setattr(main, "get_corporate_actions", lambda ticker: _fake_corporate_actions())
    client.post("/v1/watchlist", json={"ticker": "AAPL"}, headers=auth_headers)

    resp = client.get("/v1/watchlist", headers=auth_headers)
    item = resp.json()["items"][0]
    assert item["next_earnings_date"] == "2026-10-30"
    assert item["next_ex_dividend_date"] == "2026-08-10"
    assert item["last_dividend_amount"] == 0.27
    assert item["last_split"] == {"date": "2020-08-31", "ratio": 4.0}


def test_a_failed_corporate_actions_lookup_does_not_500_or_drop_the_price(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/watchlist", json={"ticker": "AAPL"}, headers=auth_headers)

    def raise_error(ticker):
        raise Exception("yfinance hiccup")

    monkeypatch.setattr(main, "get_corporate_actions", raise_error)

    resp = client.get("/v1/watchlist", headers=auth_headers)
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["price"] == 200.0
    assert item["next_earnings_date"] is None
    assert item["last_split"] is None


# ---------------------------------------------------------------- last-known rating

def test_watchlist_rating_reflects_the_users_last_completed_report(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())
    client.post("/v1/watchlist", json={"ticker": "AAPL"}, headers=auth_headers)

    # Run one real (stubbed) research job for AAPL -> mark_done should
    # denormalize ticker/rating onto the jobs row.
    submit_resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    job_id = submit_resp.json()["job_id"]

    import time
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if client.get(f"/v1/research/{job_id}", headers=auth_headers).json()["status"] == db.STATUS_DONE:
            break
        time.sleep(0.05)

    resp = client.get("/v1/watchlist", headers=auth_headers)
    items = {item["ticker"]: item for item in resp.json()["items"]}
    assert items["AAPL"]["rating"] == "Buy"


# ---------------------------------------------------------------- company suggest

def test_suggest_companies_endpoint_returns_matches(client, monkeypatch, auth_headers):
    monkeypatch.setattr(
        main, "suggest_companies",
        lambda q, limit=8: [{"ticker": "TATASTEEL.NS", "name": "Tata Steel Limited"}],
    )

    resp = client.get("/v1/companies/suggest?q=tata", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["suggestions"] == [{"ticker": "TATASTEEL.NS", "name": "Tata Steel Limited"}]


def test_suggest_companies_endpoint_requires_a_session(client):
    resp = client.get("/v1/companies/suggest?q=tata")
    assert resp.status_code == 401


def test_suggest_companies_endpoint_defaults_to_empty_query(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "suggest_companies", lambda q, limit=8: [] if not q else [{"ticker": "X", "name": "X"}])
    resp = client.get("/v1/companies/suggest", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["suggestions"] == []


# ---------------------------------------------------------------- market news

def test_market_news_returns_articles(client, monkeypatch, auth_headers):
    fake_articles = [{"headline": "Market rallies", "source": "Test Wire", "date": "2026-08-06", "url": "https://example.com/1", "summary": ""}]
    monkeypatch.setattr(main, "fetch_market_news", lambda limit=15: fake_articles)

    resp = client.get("/v1/news/market", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["articles"] == fake_articles


def test_market_news_requires_a_session(client):
    resp = client.get("/v1/news/market")
    assert resp.status_code == 401


def test_market_news_passes_through_the_limit(client, monkeypatch, auth_headers):
    captured = {}

    def fake_fetch(limit=15):
        captured["limit"] = limit
        return []

    monkeypatch.setattr(main, "fetch_market_news", fake_fetch)
    client.get("/v1/news/market?limit=5", headers=auth_headers)
    assert captured["limit"] == 5


def test_market_news_caps_the_limit_at_30(client, auth_headers):
    resp = client.get("/v1/news/market?limit=100", headers=auth_headers)
    assert resp.status_code == 422


# ---------------------------------------------------------------- market indices

def test_market_indices_returns_the_curated_list(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_quote", lambda ticker: _fake_quote())

    resp = client.get("/v1/market/indices", headers=auth_headers)
    assert resp.status_code == 200
    indices = resp.json()["indices"]
    assert [i["ticker"] for i in indices] == [e["ticker"] for e in main.INDEX_LIST]
    assert all(i["price"] == 200.0 and i["change_pct"] == 1.5 for i in indices)


def test_market_indices_requires_a_session(client):
    resp = client.get("/v1/market/indices")
    assert resp.status_code == 401


def test_one_bad_index_does_not_500_the_whole_carousel(client, monkeypatch, auth_headers):
    def flaky_quote(ticker):
        if ticker == "^BSESN":
            raise TickerNotFoundError(ticker)
        return _fake_quote()

    monkeypatch.setattr(main, "get_quote", flaky_quote)

    resp = client.get("/v1/market/indices", headers=auth_headers)
    assert resp.status_code == 200
    indices = {i["ticker"]: i for i in resp.json()["indices"]}
    assert indices["^BSESN"]["price"] is None
    assert indices["^BSESN"]["change_pct"] is None
    assert indices["^GSPC"]["price"] == 200.0


# ---------------------------------------------------------------- migration idempotency

def test_init_db_migration_is_idempotent_against_a_pre_migration_schema(tmp_path, monkeypatch):
    """Simulates the real live Cloud Run DB: a jobs table that predates
    the ticker/company_name/rating columns. init_db() must add them
    without raising, and calling it a second time (e.g. a second
    container restart) must not raise either."""
    import sqlite3

    db_path = tmp_path / "pre_migration.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL DEFAULT 'anonymous',
            status TEXT NOT NULL,
            question TEXT NOT NULL,
            orchestrator TEXT NOT NULL,
            result_json TEXT,
            pdf_path TEXT,
            error_code TEXT,
            error_message TEXT,
            started_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db, "DB_PATH", db_path)

    db.init_db()
    db.init_db()  # second call must also be a no-op, not a re-raise

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    conn.close()
    assert {"ticker", "company_name", "rating"}.issubset(columns)
