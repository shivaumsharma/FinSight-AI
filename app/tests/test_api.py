"""
Unit/integration tests for the app/api/ FastAPI service.

Uses a stub agent (same dependency-injection spirit as
test_langgraph_agent.py's stub tools) via jobs.ORCHESTRATORS, monkeypatched
per test -- so these run fast with no real network/model calls, while
still exercising the real HTTP layer, the real background-thread job
runner, and the real SQLite persistence (pointed at a per-test temp file).
"""

import io
import threading
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
from app.data import sarvam_client
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


# ---------------------------------------------------------------- stock overview

def test_get_stock_overview_returns_the_full_shape(client, monkeypatch, auth_headers):
    fake_overview = {
        "ticker": "AAPL", "price": 227.5, "change_pct": 1.2, "previous_close": 224.8,
        "currency": "USD", "company_name": "Apple Inc.", "sector": "Technology",
        "industry": "Consumer Electronics", "market_cap": 3_400_000_000_000,
        "business_summary": "Apple designs...", "website": "https://apple.com",
        "employees": 164000, "country": "United States", "exchange": "NMS",
        "next_earnings_date": None,
        "price_statistics": {"open": 225.0, "day_high": 228.1, "day_low": 224.9,
                              "fifty_two_week_high": 260.1, "fifty_two_week_low": 164.1,
                              "volume": 50_000_000, "average_volume": 55_000_000},
        "fundamentals": {"trailing_pe": 34.2, "forward_pe": 30.1, "price_to_book": 45.3,
                          "dividend_yield": 0.5, "book_value": 5.1, "debt_to_equity": 150.2,
                          "trailing_eps": 6.6, "peg_ratio": 2.1},
        "analyst": {"target_mean_price": 240.0, "target_high_price": 280.0,
                    "target_low_price": 200.0, "number_of_analyst_opinions": 42},
    }
    monkeypatch.setattr(main, "get_stock_overview", lambda ticker: fake_overview)

    resp = client.get("/v1/stocks/AAPL/overview", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == fake_overview


def test_get_stock_overview_requires_auth(client, monkeypatch):
    monkeypatch.setattr(main, "get_stock_overview", lambda ticker: {})
    resp = client.get("/v1/stocks/AAPL/overview")
    assert resp.status_code == 401


def test_get_stock_overview_returns_404_shaped_error_for_bad_ticker(client, monkeypatch, auth_headers):
    def _raise(ticker):
        raise TickerNotFoundError(f"No price data found for {ticker}")

    monkeypatch.setattr(main, "get_stock_overview", _raise)
    monkeypatch.setattr(main, "resolve_companies", lambda q: [])

    resp = client.get("/v1/stocks/ZZZZ/overview", headers=auth_headers)

    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


def test_get_stock_overview_falls_back_to_company_name_resolution(client, monkeypatch, auth_headers):
    # A hand-typed company name (e.g. from a URL) should retry once
    # through resolve_companies, same fallback the watchlist/portfolio/
    # orders POST endpoints already use.
    calls = []

    def _get_overview(ticker):
        calls.append(ticker)
        if ticker == "apple inc":
            raise TickerNotFoundError("bad")
        return {"ticker": ticker}

    monkeypatch.setattr(main, "get_stock_overview", _get_overview)
    monkeypatch.setattr(main, "resolve_companies", lambda q: ["AAPL"])

    resp = client.get("/v1/stocks/apple%20inc/overview", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == {"ticker": "AAPL"}
    assert calls == ["apple inc", "AAPL"]


def test_get_stock_financials_returns_series_and_growth(client, monkeypatch, auth_headers):
    fake_series = [{"period_end": "2024-12-31", "revenue": 100.0, "net_income": 10.0,
                     "ebit": 20.0, "net_margin_pct": 10.0, "operating_margin_pct": 20.0}]
    fake_growth = {"revenue_cagr": {"1y": 10.0, "3y": None, "5y": None},
                    "eps_cagr": {"1y": None, "3y": None, "5y": None},
                    "book_value_cagr": {"1y": None, "3y": None, "5y": None},
                    "fcf_cagr": {"1y": None, "3y": None, "5y": None}}
    monkeypatch.setattr(main, "build_financial_performance", lambda ticker, period: fake_series)
    monkeypatch.setattr(main, "build_growth_metrics", lambda ticker: fake_growth)

    resp = client.get("/v1/stocks/AAPL/financials?period=quarterly", headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["period"] == "quarterly"
    assert body["series"] == fake_series
    assert body["growth"] == fake_growth


def test_get_stock_financials_defaults_to_yearly(client, monkeypatch, auth_headers):
    captured = {}

    def _fake_series(ticker, period):
        captured["period"] = period
        return []

    monkeypatch.setattr(main, "build_financial_performance", _fake_series)
    monkeypatch.setattr(main, "build_growth_metrics", lambda ticker: {})

    resp = client.get("/v1/stocks/AAPL/financials", headers=auth_headers)

    assert resp.status_code == 200
    assert captured["period"] == "yearly"


def test_get_stock_financials_requires_auth(client):
    resp = client.get("/v1/stocks/AAPL/financials")
    assert resp.status_code == 401


def test_get_stock_financials_maps_missing_statements_to_ticker_not_found(client, monkeypatch, auth_headers):
    def _raise(ticker, period):
        raise ValueError("Income Statement unavailable")

    monkeypatch.setattr(main, "build_financial_performance", _raise)

    resp = client.get("/v1/stocks/ZZZZ/financials", headers=auth_headers)

    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


def test_get_stock_financials_degrades_growth_to_empty_on_failure(client, monkeypatch, auth_headers):
    # A ticker can have a usable income statement (series succeeds) but
    # a balance-sheet/cash-flow gap that breaks growth CAGR -- the
    # series result must still be returned rather than the whole
    # endpoint failing.
    monkeypatch.setattr(main, "build_financial_performance", lambda ticker, period: [{"period_end": "2024-12-31"}])

    def _raise_growth(ticker):
        raise ValueError("Balance Sheet unavailable")

    monkeypatch.setattr(main, "build_growth_metrics", _raise_growth)

    resp = client.get("/v1/stocks/AAPL/financials", headers=auth_headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["series"] == [{"period_end": "2024-12-31"}]
    assert body["growth"] == {"revenue_cagr": {}, "eps_cagr": {}, "book_value_cagr": {}, "fcf_cagr": {}}


# ---------------------------------------------------------------- stock technicals

def test_get_stock_technicals_returns_the_full_shape(client, monkeypatch, auth_headers):
    fake = {
        "price_history": [{"date": "2024-01-01", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000}],
        "overlays": {"ema_20": [["2024-01-01", 100.0]], "ema_50": [], "ema_200": [], "bollinger_upper": [], "bollinger_lower": [], "vwap": []},
        "indicators": {"rsi_14": 55.0, "macd": {"macd": 1.0, "signal": 0.5, "histogram": 0.5}, "adx_14": 20.0,
                        "vwap": 100.0, "ema_20": 100.0, "ema_50": 99.0, "ema_200": 95.0,
                        "stochastic": {"k": 60.0, "d": 55.0}, "williams_r": -40.0},
        "trend": "Bullish",
        "moving_average_signal": {"buy": 5, "sell": 1, "verdict": "Buy"},
        "support_resistance": {"support_20d": 95.0, "resistance_20d": 105.0, "support_60d": 90.0, "resistance_60d": 110.0},
    }
    monkeypatch.setattr(main, "build_technicals", lambda ticker, range_period: fake)

    resp = client.get("/v1/stocks/AAPL/technicals?range=1y", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == fake


def test_get_stock_technicals_defaults_to_1y_range(client, monkeypatch, auth_headers):
    captured = {}

    def _fake(ticker, range_period):
        captured["range_period"] = range_period
        return {}

    monkeypatch.setattr(main, "build_technicals", _fake)

    resp = client.get("/v1/stocks/AAPL/technicals", headers=auth_headers)

    assert resp.status_code == 200
    assert captured["range_period"] == "1y"


def test_get_stock_technicals_rejects_an_invalid_range(client, auth_headers):
    resp = client.get("/v1/stocks/AAPL/technicals?range=10y", headers=auth_headers)
    assert resp.status_code == 422  # FastAPI's own Literal-validation error, not a hand-rolled one


def test_get_stock_technicals_requires_auth(client):
    resp = client.get("/v1/stocks/AAPL/technicals")
    assert resp.status_code == 401


def test_get_stock_technicals_maps_bad_ticker_to_ticker_not_found(client, monkeypatch, auth_headers):
    def _raise(ticker, range_period):
        raise ValueError("No price data found")

    monkeypatch.setattr(main, "build_technicals", _raise)

    resp = client.get("/v1/stocks/ZZZZ/technicals", headers=auth_headers)

    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


# ---------------------------------------------------------------- stock insights

def test_get_stock_insights_returns_the_full_shape(client, monkeypatch, auth_headers):
    fake = {
        "rating": "Buy", "overall_score": 78,
        "category_scores": {"valuation": 4, "financial_health": 5, "growth": 3, "profitability": 4, "momentum": 5, "risk": 4},
        "fair_value_estimate": 250.0, "current_price": 200.0, "upside_percent": 25.0,
        "consensus": {"score": 80, "label": "Strong Agreement"},
        "flags": ["Strong Financials", "Positive Momentum"],
    }
    monkeypatch.setattr(main, "build_stock_insights", lambda ticker: fake)

    resp = client.get("/v1/stocks/AAPL/insights", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == fake


def test_get_stock_insights_requires_auth(client):
    resp = client.get("/v1/stocks/AAPL/insights")
    assert resp.status_code == 401


def test_get_stock_insights_maps_bad_ticker_to_ticker_not_found(client, monkeypatch, auth_headers):
    def _raise(ticker):
        raise TickerNotFoundError("bad ticker")

    monkeypatch.setattr(main, "build_stock_insights", _raise)

    resp = client.get("/v1/stocks/ZZZZ/insights", headers=auth_headers)

    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


def test_stock_portfolio_fit_returns_the_fit_summary(client, monkeypatch, auth_headers):
    fake = {"sector": "Technology", "current_allocation_pct": 42.0, "summary": "You're already 42% in Technology."}
    monkeypatch.setattr(main, "get_portfolio_fit_for_ticker", lambda ticker, holdings: fake)

    resp = client.get("/v1/stocks/AAPL/portfolio-fit", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == fake


def test_stock_portfolio_fit_requires_auth(client):
    resp = client.get("/v1/stocks/AAPL/portfolio-fit")
    assert resp.status_code == 401


def test_stock_model_compare_returns_models_and_consensus(client, monkeypatch, auth_headers):
    fake = {
        "models": [{"label": "Llama 3.3 70B", "model": "llama-3.3-70b-versatile", "rating": "Buy", "confidence": 70, "reasoning": "..."}],
        "consensus": {"rating": "Buy", "agree_count": 1, "total": 1},
    }
    monkeypatch.setattr(main, "get_stock_model_opinions", lambda ticker: fake)

    resp = client.post("/v1/stocks/AAPL/model-compare", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == fake


def test_stock_model_compare_requires_auth(client):
    resp = client.post("/v1/stocks/AAPL/model-compare")
    assert resp.status_code == 401


def test_stock_model_compare_maps_bad_ticker_to_ticker_not_found(client, monkeypatch, auth_headers):
    def _raise(ticker):
        raise TickerNotFoundError("bad ticker")

    monkeypatch.setattr(main, "get_stock_model_opinions", _raise)

    resp = client.post("/v1/stocks/ZZZZ/model-compare", headers=auth_headers)

    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


# ---------------------------------------------------------------- stock news/events/similar/peer-comparison

def test_get_stock_news_returns_articles(client, monkeypatch, auth_headers):
    fake_articles = [{"headline": "Apple beats earnings", "source": "Reuters", "date": "2026-08-01",
                       "url": "https://example.com/a", "summary": "...", "categories": ["other"]}]
    monkeypatch.setattr(main, "fetch_company_news", lambda ticker: fake_articles)

    resp = client.get("/v1/stocks/AAPL/news", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == {"articles": fake_articles}


def test_get_stock_news_requires_auth(client):
    resp = client.get("/v1/stocks/AAPL/news")
    assert resp.status_code == 401


def test_get_stock_events_returns_corporate_actions_history(client, monkeypatch, auth_headers):
    fake = {"next_earnings_date": "2026-09-01", "next_ex_dividend_date": None,
            "dividends": [{"date": "2025-01-01", "amount": 0.24}], "splits": []}
    monkeypatch.setattr(main, "get_corporate_actions_history", lambda ticker: fake)

    resp = client.get("/v1/stocks/AAPL/events", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == fake


def test_get_similar_stocks_returns_sector_and_matches(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_stock_overview", lambda ticker: {"sector": "Technology"})
    monkeypatch.setattr(main, "find_similar_stocks", lambda ticker, sector: [{"ticker": "MSFT"}])

    resp = client.get("/v1/stocks/AAPL/similar", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == {"sector": "Technology", "similar": [{"ticker": "MSFT"}]}


def test_get_similar_stocks_maps_bad_ticker_to_ticker_not_found(client, monkeypatch, auth_headers):
    def _raise(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(main, "get_stock_overview", _raise)

    resp = client.get("/v1/stocks/ZZZZ/similar", headers=auth_headers)

    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


def test_get_peer_comparison_returns_primary_and_peer(client, monkeypatch, auth_headers):
    fake = {"primary": {"ticker": "AAPL"}, "peer": {"ticker": "MSFT"}}
    monkeypatch.setattr(main, "build_peer_comparison", lambda ticker, peer: fake)

    resp = client.get("/v1/stocks/AAPL/peer-comparison?peer=MSFT", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json() == fake


def test_get_peer_comparison_requires_peer_query_param(client, auth_headers):
    resp = client.get("/v1/stocks/AAPL/peer-comparison", headers=auth_headers)
    assert resp.status_code == 422


def test_get_peer_comparison_maps_bad_ticker_to_ticker_not_found(client, monkeypatch, auth_headers):
    def _raise(ticker, peer):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(main, "build_peer_comparison", _raise)

    resp = client.get("/v1/stocks/AAPL/peer-comparison?peer=ZZZZ", headers=auth_headers)

    assert resp.status_code == 400
    assert resp.json()["code"] == "TICKER_NOT_FOUND"


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


# ---------------------------------------------------------------- backtest accuracy badge

def test_backtest_accuracy_returns_the_real_computed_summary(client, monkeypatch, auth_headers):
    fake_summary = {"accuracy_pct": 52.6, "correct": 41, "scored": 78, "window_label": "x", "secondary": None}
    monkeypatch.setattr(main, "get_backtest_accuracy_summary", lambda: fake_summary)

    resp = client.get("/v1/research/backtest-accuracy", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == fake_summary


def test_backtest_accuracy_404s_when_unavailable(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_backtest_accuracy_summary", lambda: None)

    resp = client.get("/v1/research/backtest-accuracy", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["code"] == "BACKTEST_ACCURACY_UNAVAILABLE"


def test_backtest_accuracy_requires_a_session(client):
    resp = client.get("/v1/research/backtest-accuracy")
    assert resp.status_code == 401


def test_backtest_accuracy_is_registered_before_the_job_id_route(client, auth_headers):
    # Same Starlette route-ordering hazard as /v1/research/recent above
    # -- "backtest-accuracy" must not get matched as a literal job_id.
    resp = client.get("/v1/research/backtest-accuracy", headers=auth_headers)
    assert resp.status_code in (200, 404)
    assert resp.json().get("code") != "JOB_NOT_FOUND"


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


def _onboarding_body(**overrides):
    body = {
        "risk_tolerance": "Aggressive",
        "investment_goal": "Wealth Growth",
        "investment_horizon": "Long-term (7y+)",
        "interested_in_crypto": True,
        "interested_in_real_estate": False,
    }
    body.update(overrides)
    return body


def test_onboarding_defaults_to_not_completed(client, auth_headers):
    resp = client.get("/v1/auth/me", headers=auth_headers)
    data = resp.json()
    assert data["onboarding_completed"] is False
    assert data["investment_goal"] is None
    assert data["interested_in_crypto"] is False


def test_onboarding_persists_all_fields_and_flips_completed(client, auth_headers):
    resp = client.patch("/v1/auth/onboarding", json=_onboarding_body(), headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get("/v1/auth/me", headers=auth_headers)
    data = resp.json()
    assert data["onboarding_completed"] is True
    assert data["risk_tolerance"] == "Aggressive"
    assert data["investment_goal"] == "Wealth Growth"
    assert data["investment_horizon"] == "Long-term (7y+)"
    assert data["interested_in_crypto"] is True
    assert data["interested_in_real_estate"] is False


def test_onboarding_rejects_an_invalid_goal(client, auth_headers):
    resp = client.patch("/v1/auth/onboarding", json=_onboarding_body(investment_goal="Get Rich Quick"), headers=auth_headers)
    assert resp.status_code == 422


def test_onboarding_rejects_an_invalid_horizon(client, auth_headers):
    resp = client.patch("/v1/auth/onboarding", json=_onboarding_body(investment_horizon="Forever"), headers=auth_headers)
    assert resp.status_code == 422


def test_onboarding_requires_auth(client):
    resp = client.patch("/v1/auth/onboarding", json=_onboarding_body())
    assert resp.status_code == 401


def test_real_estate_guidance_is_empty_when_not_opted_in(client, auth_headers):
    resp = client.get("/v1/auth/real-estate-guidance", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == {}


def test_real_estate_guidance_reflects_onboarding_answers(client, auth_headers):
    client.patch("/v1/auth/onboarding", json=_onboarding_body(interested_in_real_estate=True, risk_tolerance="Conservative"), headers=auth_headers)

    resp = client.get("/v1/auth/real-estate-guidance", headers=auth_headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["target_allocation_low_pct"] == 15.0
    assert data["example_reit_tickers"] == ["VNQ", "O", "PLD"]


def test_real_estate_guidance_requires_auth(client):
    resp = client.get("/v1/auth/real-estate-guidance")
    assert resp.status_code == 401


def test_display_name_defaults_to_none_and_is_persisted(client, auth_headers):
    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["display_name"] is None

    resp = client.patch("/v1/auth/display-name", json={"display_name": "Shivaum"}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "Shivaum"

    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["display_name"] == "Shivaum"


def test_display_name_is_trimmed(client, auth_headers):
    resp = client.patch("/v1/auth/display-name", json={"display_name": "  Shivaum  "}, headers=auth_headers)
    assert resp.json()["display_name"] == "Shivaum"


def test_display_name_of_empty_or_whitespace_clears_it_back_to_none(client, auth_headers):
    client.patch("/v1/auth/display-name", json={"display_name": "Shivaum"}, headers=auth_headers)
    resp = client.patch("/v1/auth/display-name", json={"display_name": "   "}, headers=auth_headers)
    assert resp.json()["display_name"] is None

    resp = client.get("/v1/auth/me", headers=auth_headers)
    assert resp.json()["display_name"] is None


def test_display_name_rejects_more_than_40_characters(client, auth_headers):
    resp = client.patch("/v1/auth/display-name", json={"display_name": "x" * 41}, headers=auth_headers)
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


# ---------------------------------------------------------------- model compare

def _fake_opinions(*_a, **_k):
    return [
        {"label": "Llama 3.3 70B", "model": "llama-3.3-70b-versatile", "rating": "Buy", "confidence": 78, "reasoning": "Upside supports it."},
        {"label": "GPT-OSS 120B", "model": "openai/gpt-oss-120b", "rating": "Buy", "confidence": 82, "reasoning": "Sentiment agrees."},
        {"label": "Qwen3.6 27B", "model": "qwen/qwen3.6-27b", "rating": "Hold", "confidence": 55, "reasoning": "Wants more data."},
    ]


def test_model_compare_returns_models_and_consensus(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "get_model_opinions", _fake_opinions)
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers)

    resp = client.post(f"/v1/research/{job_id}/model-compare", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["models"]) == 3
    assert body["consensus"] == {"rating": "Buy", "agree_count": 2, "total": 3}


def test_model_compare_requires_a_session(client):
    resp = client.post("/v1/research/some-job-id/model-compare")
    assert resp.status_code == 401


def test_model_compare_404s_for_an_unknown_job(client, auth_headers):
    resp = client.post("/v1/research/does-not-exist/model-compare", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["code"] == "JOB_NOT_FOUND"


def test_model_compare_only_the_owner_can_run_it(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]

    other_user_headers = _signup(client, email="mallory-compare@example.com", password="malloryspassword")
    resp = client.post(f"/v1/research/{job_id}/model-compare", headers=other_user_headers)
    assert resp.status_code == 403
    assert resp.json()["code"] == "FORBIDDEN"


def test_model_compare_409s_while_job_is_not_done(client, auth_headers):
    job_id = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers
    ).json()["job_id"]
    resp = client.post(f"/v1/research/{job_id}/model-compare", headers=auth_headers)
    # Could race with the (fast) stub finishing first -- only assert the
    # 409 case when it's actually still pending/running.
    status = client.get(f"/v1/research/{job_id}", headers=auth_headers).json()["status"]
    if status in (db.STATUS_PENDING, db.STATUS_RUNNING):
        assert resp.status_code == 409
        assert resp.json()["code"] == "JOB_NOT_DONE"


def test_exceeding_the_daily_job_limit_is_rejected(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "DAILY_JOB_LIMIT", 2)

    # Distinct question text per request, deliberately -- an identical
    # repeat would hit find_recent_duplicate_job's reuse path (see
    # test_duplicate_request_reuses_existing_job below, which tests that
    # path directly) and never consume quota at all, which is a
    # different mechanism from the one this test exercises.
    #
    # Polled to completion here (not left running) -- jobs.py's executor
    # is a single module-level ThreadPoolExecutor shared by every test in
    # the process, not recreated per test. A job left queued past this
    # test's own teardown would get picked up later with this test's
    # monkeypatched ORCHESTRATORS/LLM_PROVIDER already reverted, i.e. it
    # would run the real orchestrator instead of _StubAgent -- slow (or
    # network-dependent) and liable to starve whatever test runs next
    # while it sits at the front of that single-worker queue.
    for i in range(2):
        resp = client.post(
            "/v1/research", json={"question": f"Should I invest in AAPL? (request {i})"}, headers=auth_headers
        )
        assert resp.status_code == 200
        _poll_until_terminal(client, resp.json()["job_id"], auth_headers, timeout=15.0)

    resp = client.post(
        "/v1/research", json={"question": "Should I invest in AAPL? (request 2)"}, headers=auth_headers
    )
    assert resp.status_code == 429
    assert resp.json()["code"] == "RATE_LIMIT_EXCEEDED"


def test_rate_limit_is_scoped_per_user_not_global(client, monkeypatch, auth_headers):
    # A second user hitting their own limit must not be affected by the
    # first user's usage -- count_recent_jobs is keyed by user_id, not
    # a single global counter.
    monkeypatch.setattr(main, "DAILY_JOB_LIMIT", 1)

    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 200
    # Polled to completion -- see the identical note in
    # test_exceeding_the_daily_job_limit_is_rejected above (shared
    # single-worker executor across the whole test process).
    _poll_until_terminal(client, resp.json()["job_id"], auth_headers, timeout=15.0)
    # Distinct question text -- see the same note in
    # test_exceeding_the_daily_job_limit_is_rejected above; an identical
    # repeat would hit the duplicate-reuse path instead of the quota path.
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL? (again)"}, headers=auth_headers)
    assert resp.status_code == 429

    other_user_headers = _signup(client, email="ninacountsseparately@example.com", password="ninaspassword")
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=other_user_headers)
    assert resp.status_code == 200
    _poll_until_terminal(client, resp.json()["job_id"], other_user_headers, timeout=15.0)


def test_create_job_if_under_limit_boundary_sequential(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    db.init_db()

    window_start = time.time() - 3600
    for i in range(3):
        job_id = db.create_job_if_under_limit(
            question=f"q{i}", orchestrator="hand_rolled", user_id="alice",
            limit=3, window_start=window_start,
        )
        assert job_id is not None

    # The 4th call is over the limit -- no row created, quota untouched.
    assert db.create_job_if_under_limit(
        question="q3", orchestrator="hand_rolled", user_id="alice",
        limit=3, window_start=window_start,
    ) is None
    assert db.count_recent_jobs("alice", window_start) == 3


def test_create_job_if_under_limit_is_race_free_under_concurrency(tmp_path, monkeypatch):
    # The one genuinely novel test pattern in this suite -- see
    # create_job_if_under_limit's own docstring for the check-then-act
    # race this closes (two concurrent callers both reading the same
    # pre-insert COUNT). Sequential testing (the test above) cannot
    # expose a regression back to that race; only real concurrent
    # callers hitting the same connection-per-call SQLite file can.
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    db.init_db()

    window_start = time.time() - 3600
    limit = 5
    n_threads = 20
    results = [None] * n_threads

    def _attempt(i):
        results[i] = db.create_job_if_under_limit(
            question=f"concurrent {i}", orchestrator="hand_rolled", user_id="bob",
            limit=limit, window_start=window_start,
        )

    threads = [threading.Thread(target=_attempt, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    succeeded = [r for r in results if r is not None]
    assert len(succeeded) == limit
    assert len(set(succeeded)) == limit  # every success got its own distinct job_id
    assert db.count_recent_jobs("bob", window_start) == limit


def test_duplicate_request_reuses_existing_job(client, auth_headers):
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    # 15s, not the 5s default -- see test_portfolio.py/test_watchlist.py's
    # identical polling loops (and 26c55a2's commit message) for why:
    # shared/loaded runners are consistently slower than a quiet dev
    # machine, and this exact class of test was caught flaking on both.
    _poll_until_terminal(client, job_id, auth_headers, timeout=15.0)

    used_before = client.get("/v1/auth/me", headers=auth_headers).json()["jobs_used_today"]
    # Same user, same exact question, while the prior job is still
    # within the duplicate window and DONE -- reuses it instead of
    # spending another quota unit re-running the pipeline.
    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["reused"] is True
    used_after = client.get("/v1/auth/me", headers=auth_headers).json()["jobs_used_today"]
    assert used_after == used_before


def test_duplicate_request_outside_window_creates_new_job(client, monkeypatch, auth_headers):
    # Window of 0 means "since" is effectively now -- the prior job's
    # started_at (in the past) no longer satisfies started_at > since,
    # so it must not be treated as a reusable duplicate.
    monkeypatch.setattr(main, "DUPLICATE_REQUEST_WINDOW_SECONDS", 0)

    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    _poll_until_terminal(client, job_id, auth_headers, timeout=15.0)

    resp = client.post("/v1/research", json={"question": "Should I invest in AAPL?"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] != job_id
    assert "reused" not in body


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


def test_unhandled_exception_returns_structured_internal_error(client, monkeypatch, auth_headers):
    # /v1/companies/suggest has no try/except of its own -- a real bug
    # here (or anywhere else lacking one) would otherwise leak
    # Starlette's raw unhandled-exception response instead of this API's
    # normal {code, message} shape.
    #
    # A dedicated TestClient with raise_server_exceptions=False -- the
    # default client fixture leaves that True, which makes Starlette's
    # ServerErrorMiddleware re-raise into the test process even after
    # invoking a registered handler and building its response (by
    # design, so test suites can see the traceback); that would defeat
    # the point of this test, which is to inspect the actual HTTP
    # response a real caller gets back.
    def _boom(q, limit=8):
        raise RuntimeError("synthetic failure for the generic handler test")

    monkeypatch.setattr(main, "suggest_companies", _boom)
    no_raise_client = TestClient(app, raise_server_exceptions=False)
    resp = no_raise_client.get("/v1/companies/suggest?q=tata", headers=auth_headers)
    assert resp.status_code == 500
    assert resp.json()["code"] == "INTERNAL_ERROR"


def test_api_error_still_routes_to_its_own_handler_not_the_generic_one(client, auth_headers):
    # FastAPI resolves exception handlers by most-specific registered
    # type -- errors.APIError is a subclass of Exception, so this must
    # keep hitting api_error_handler (its real status code/code/message)
    # rather than falling through to the generic 500/INTERNAL_ERROR path
    # now that both are registered.
    resp = client.get("/v1/research/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404
    assert resp.json()["code"] == "JOB_NOT_FOUND"


# ---------------------------------------------------------- POST /v1/voice/transcribe

def test_transcribe_voice_success(client, monkeypatch, auth_headers):
    monkeypatch.setattr(sarvam_client, "transcribe", lambda *a, **k: "Should I invest in AAPL?")

    resp = client.post(
        "/v1/voice/transcribe",
        files={"file": ("recording.webm", b"fake-audio-bytes", "audio/webm")},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert resp.json() == {"transcript": "Should I invest in AAPL?"}


def test_transcribe_voice_requires_auth(client):
    resp = client.post(
        "/v1/voice/transcribe",
        files={"file": ("recording.webm", b"fake-audio-bytes", "audio/webm")},
    )
    assert resp.status_code == 401


def test_transcribe_voice_maps_sarvam_failure_to_503(client, monkeypatch, auth_headers):
    def _boom(*a, **k):
        raise sarvam_client.SarvamTranscriptionError("Sarvam STT request failed: synthetic")

    monkeypatch.setattr(sarvam_client, "transcribe", _boom)

    resp = client.post(
        "/v1/voice/transcribe",
        files={"file": ("recording.webm", b"fake-audio-bytes", "audio/webm")},
        headers=auth_headers,
    )

    assert resp.status_code == 503
    assert resp.json()["code"] == "STT_UNAVAILABLE"


def test_transcribe_voice_rejects_empty_file(client, auth_headers):
    resp = client.post(
        "/v1/voice/transcribe",
        files={"file": ("recording.webm", b"", "audio/webm")},
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_AUDIO"


def test_transcribe_voice_rejects_oversized_file(client, monkeypatch, auth_headers):
    monkeypatch.setattr(main, "MAX_VOICE_AUDIO_BYTES", 10)

    resp = client.post(
        "/v1/voice/transcribe",
        files={"file": ("recording.webm", b"x" * 100, "audio/webm")},
        headers=auth_headers,
    )

    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_AUDIO"
