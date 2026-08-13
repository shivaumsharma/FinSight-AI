"""
Unit tests for app/reasoning/chat_router.py. HostedProvider is
monkeypatched to a fake class per test (same convention as
test_model_consensus.py) -- never hits real network.
"""

import pytest

from app.core.llm_provider import LLMProviderError
from app.data.market_data import TickerNotFoundError
from app.reasoning import chat_router as cr


class _FakeProvider:
    def __init__(self, model=None, **kwargs):
        self.model = model

    def generate(self, prompt, max_new_tokens=150):
        return "INTENT: portfolio_status"


class _RaisingProvider:
    def __init__(self, model=None, **kwargs):
        pass

    def generate(self, prompt, max_new_tokens=150):
        raise LLMProviderError("simulated outage")


# ---------------------------------------------------------------- _parse_intent

def test_parse_intent_extracts_a_valid_label():
    assert cr._parse_intent("INTENT: ticker_question") == "ticker_question"


def test_parse_intent_is_case_insensitive():
    assert cr._parse_intent("intent: PORTFOLIO_STATUS") == "portfolio_status"


def test_parse_intent_degrades_to_general_on_garbage():
    assert cr._parse_intent("I refuse to classify this.") == "general"


def test_parse_intent_degrades_to_general_on_an_unknown_label():
    assert cr._parse_intent("INTENT: banana") == "general"


# ---------------------------------------------------------------- classify_intent

def test_classify_intent_uses_resolve_companies_for_the_ticker(monkeypatch):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["AAPL"])
    monkeypatch.setattr(cr, "HostedProvider", _FakeProvider)

    result = cr.classify_intent("what's going on with apple")

    assert result == {"intent": "portfolio_status", "ticker": "AAPL"}


def test_classify_intent_no_ticker_detected(monkeypatch):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _FakeProvider)

    result = cr.classify_intent("hows my portfolio")

    assert result["ticker"] is None


def test_classify_intent_degrades_ticker_scoped_intent_with_no_ticker(monkeypatch):
    class _TickerQuestionProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return "INTENT: ticker_question"

    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _TickerQuestionProvider)

    result = cr.classify_intent("what's a good stock to buy")

    assert result["intent"] == "general"


def test_classify_intent_falls_back_to_general_on_llm_outage(monkeypatch):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)

    result = cr.classify_intent("anything")

    assert result["intent"] == "general"


# ---------------------------------------------------------------- _handle_portfolio_status

def test_handle_portfolio_status_with_no_holdings(monkeypatch):
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"holdings": [], "summary": {}})
    assert "don't have any holdings" in cr._handle_portfolio_status("u1", "hows my portfolio")


def test_handle_portfolio_status_grounds_the_llm_in_real_holdings_and_the_actual_question(monkeypatch):
    view = {
        "holdings": [
            {"ticker": "AAPL", "quantity": 4, "price": 200.0, "currency": "USD", "rating": "Buy"},
        ],
        "summary": {"total_market_value": 800.0, "total_unrealized_pnl": 50.0, "total_unrealized_pnl_pct": 6.67},
    }
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: view)
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [{"ticker": "AAPL", "quantity": 4}])
    monkeypatch.setattr(cr, "get_portfolio_sector_allocation", lambda holdings: {"Technology": 100.0})

    captured = {}

    class _StatusProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            captured["prompt"] = prompt
            return "You're 100% in Technology, so no, not diversified."

    monkeypatch.setattr(cr, "HostedProvider", _StatusProvider)

    result = cr._handle_portfolio_status("u1", "are they diversified enough")

    assert "not diversified" in result
    assert "AAPL" in captured["prompt"]
    assert "rated Buy" in captured["prompt"]
    assert "800.00" in captured["prompt"]
    assert "Technology 100" in captured["prompt"]
    assert "are they diversified enough" in captured["prompt"]


def test_handle_portfolio_status_falls_back_to_the_raw_summary_on_llm_outage(monkeypatch):
    view = {
        "holdings": [
            {"ticker": "AAPL", "quantity": 4, "price": 200.0, "currency": "USD", "rating": "Buy"},
        ],
        "summary": {"total_market_value": 800.0, "total_unrealized_pnl": 50.0, "total_unrealized_pnl_pct": 6.67},
    }
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: view)
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [{"ticker": "AAPL", "quantity": 4}])
    monkeypatch.setattr(cr, "get_portfolio_sector_allocation", lambda holdings: {})
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)

    result = cr._handle_portfolio_status("u1", "hows my portfolio")

    assert "1 holding" in result
    assert "AAPL" in result
    assert "rated Buy" in result
    assert "800.00" in result


# ---------------------------------------------------------------- _handle_ticker_question

def test_handle_ticker_question_summarizes_insights_and_news(monkeypatch):
    insights = {
        "rating": "Sell", "fair_value_estimate": 107.0, "current_price": 300.0,
        "upside_percent": -64.3, "flags": ["Rich Valuation"],
    }
    monkeypatch.setattr(cr, "build_stock_insights", lambda ticker: insights)
    monkeypatch.setattr(cr, "fetch_company_news", lambda ticker: [{"headline": "AAPL beats earnings"}])

    result = cr._handle_ticker_question("AAPL")

    assert "Sell" in result
    assert "107.00" in result
    assert "Rich Valuation" in result
    assert "AAPL beats earnings" in result


def test_handle_ticker_question_degrades_on_insufficient_data(monkeypatch):
    insights = {"rating": "Insufficient Data", "fair_value_estimate": None, "current_price": None, "upside_percent": None, "flags": []}
    monkeypatch.setattr(cr, "build_stock_insights", lambda ticker: insights)
    monkeypatch.setattr(cr, "fetch_company_news", lambda ticker: [])

    result = cr._handle_ticker_question("BTC-USD")

    assert "not enough data" in result


def test_handle_ticker_question_propagates_bad_ticker_as_a_plain_message(monkeypatch):
    def _raise(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(cr, "build_stock_insights", _raise)

    result = cr._handle_ticker_question("ZZZZ")

    assert "couldn't find ZZZZ" in result


def test_handle_ticker_question_survives_a_news_fetch_failure(monkeypatch):
    insights = {"rating": "Buy", "fair_value_estimate": 200.0, "current_price": 150.0, "upside_percent": 33.3, "flags": []}
    monkeypatch.setattr(cr, "build_stock_insights", lambda ticker: insights)

    def _raise_news(ticker):
        raise Exception("news feed down")

    monkeypatch.setattr(cr, "fetch_company_news", _raise_news)

    result = cr._handle_ticker_question("AAPL")

    assert "Buy" in result


# ---------------------------------------------------------------- _handle_portfolio_fit

def test_handle_portfolio_fit_delegates_to_get_portfolio_fit_for_ticker(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [{"ticker": "MSFT", "quantity": 1}])
    monkeypatch.setattr(cr, "get_portfolio_fit_for_ticker", lambda ticker, holdings: {"summary": "You're already 100% in Technology."})

    assert cr._handle_portfolio_fit("u1", "AAPL") == "You're already 100% in Technology."


# ---------------------------------------------------------------- _handle_full_report_request

def test_handle_full_report_request_mentions_the_ticker_and_the_cta():
    result = cr._handle_full_report_request("AAPL")
    assert "AAPL" in result
    assert "Run Full Research Report" in result


# ---------------------------------------------------------------- _handle_advice_request

def test_handle_advice_request_includes_onboarding_and_portfolio_context(monkeypatch):
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {
        "onboarding_completed": 1, "risk_tolerance": "Aggressive",
        "investment_goal": "Wealth Growth", "investment_horizon": "Long-term (7y+)",
        "interested_in_real_estate": 0,
    })
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [{"ticker": "AAPL", "quantity": 1}])
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {
        "holdings": [{"ticker": "AAPL"}], "summary": {"total_market_value": 500.0},
    })
    monkeypatch.setattr(cr, "get_portfolio_sector_allocation", lambda holdings: {"Technology": 100.0})

    captured = {}

    class _AdviceProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            captured["prompt"] = prompt
            return "Given your aggressive, long-term goal, consider..."

    monkeypatch.setattr(cr, "HostedProvider", _AdviceProvider)

    result = cr._handle_advice_request("u1", "where should I invest")

    assert "aggressive" in result.lower()
    assert "Aggressive" in captured["prompt"]
    assert "Technology 100" in captured["prompt"]
    assert "500.00" in captured["prompt"]


def test_handle_advice_request_with_no_onboarding_or_holdings(monkeypatch):
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {"onboarding_completed": 0})
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [])

    captured = {}

    class _AdviceProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            captured["prompt"] = prompt
            return "Start with diversified index funds..."

    monkeypatch.setattr(cr, "HostedProvider", _AdviceProvider)

    result = cr._handle_advice_request("u1", "how should I spend $40k")

    assert "index funds" in result.lower()
    assert "hasn't completed the onboarding" in captured["prompt"]
    assert "No existing portfolio holdings" in captured["prompt"]


def test_handle_advice_request_falls_back_on_llm_outage(monkeypatch):
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: None)
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [])
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)

    result = cr._handle_advice_request("u1", "where should I invest")

    assert "allocation-level guidance" in result


# ---------------------------------------------------------------- _handle_general

def test_handle_general_returns_the_llm_reply(monkeypatch):
    class _GeneralProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return "  I can help with that.  "

    monkeypatch.setattr(cr, "HostedProvider", _GeneralProvider)

    assert cr._handle_general("hello") == "I can help with that."


def test_handle_general_falls_back_on_llm_outage(monkeypatch):
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)
    result = cr._handle_general("hello")
    assert "portfolio or a specific ticker" in result


# ---------------------------------------------------------------- handle_chat_message (orchestration)

def test_handle_chat_message_dispatches_to_the_right_handler(monkeypatch):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _FakeProvider)  # returns "INTENT: portfolio_status"
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"holdings": [], "summary": {}})

    result = cr.handle_chat_message("u1", "hows my portfolio")

    assert result["intent"] == "portfolio_status"
    assert result["ticker"] is None
    assert "don't have any holdings" in result["reply"]
