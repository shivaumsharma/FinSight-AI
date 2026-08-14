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
    assert "don't have any holdings" in cr._handle_portfolio_status("u1", "hows my portfolio", [])


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

    history = [{"role": "user", "content": "what about my Apple shares"}]
    result = cr._handle_portfolio_status("u1", "are they diversified enough", history)

    assert "not diversified" in result
    assert "AAPL" in captured["prompt"]
    assert "rated Buy" in captured["prompt"]
    assert "800.00" in captured["prompt"]
    assert "Technology 100" in captured["prompt"]
    assert "are they diversified enough" in captured["prompt"]
    assert "what about my Apple shares" in captured["prompt"]


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

    result = cr._handle_portfolio_status("u1", "hows my portfolio", [])

    assert "1 holding" in result
    assert "AAPL" in result
    assert "rated Buy" in result
    assert "800.00" in result


# ---------------------------------------------------------------- _handle_ticker_question

def test_handle_ticker_question_grounds_the_llm_in_real_insights_and_the_actual_question(monkeypatch):
    insights = {
        "rating": "Sell", "fair_value_estimate": 107.0, "current_price": 300.0,
        "upside_percent": -64.3, "flags": ["Rich Valuation"],
    }
    monkeypatch.setattr(cr, "build_stock_insights", lambda ticker: insights)
    monkeypatch.setattr(cr, "fetch_company_news", lambda ticker: [{"headline": "AAPL beats earnings"}])

    captured = {}

    class _TickerProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            captured["prompt"] = prompt
            return "This is FinSight's own algorithmic rating, not a third-party call."

    monkeypatch.setattr(cr, "HostedProvider", _TickerProvider)

    history = [{"role": "assistant", "content": "AAPL: algorithmic rating is Sell.", "ticker": "AAPL"}]
    result = cr._handle_ticker_question("AAPL", "what's your source for that", history)

    assert "algorithmic rating" in result
    assert "Sell" in captured["prompt"]
    assert "107.00" in captured["prompt"]
    assert "Rich Valuation" in captured["prompt"]
    assert "AAPL beats earnings" in captured["prompt"]
    assert "what's your source for that" in captured["prompt"]


def test_handle_ticker_question_degrades_on_insufficient_data(monkeypatch):
    insights = {"rating": "Insufficient Data", "fair_value_estimate": None, "current_price": None, "upside_percent": None, "flags": []}
    monkeypatch.setattr(cr, "build_stock_insights", lambda ticker: insights)
    monkeypatch.setattr(cr, "fetch_company_news", lambda ticker: [])
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)  # exercise the fallback path

    result = cr._handle_ticker_question("BTC-USD", "what's going on with this", [])

    assert "not enough data" in result


def test_handle_ticker_question_propagates_bad_ticker_as_a_plain_message(monkeypatch):
    def _raise(ticker):
        raise TickerNotFoundError(ticker)

    monkeypatch.setattr(cr, "build_stock_insights", _raise)

    result = cr._handle_ticker_question("ZZZZ", "what's going on with this", [])

    assert "couldn't find ZZZZ" in result


def test_handle_ticker_question_survives_a_news_fetch_failure(monkeypatch):
    insights = {"rating": "Buy", "fair_value_estimate": 200.0, "current_price": 150.0, "upside_percent": 33.3, "flags": []}
    monkeypatch.setattr(cr, "build_stock_insights", lambda ticker: insights)

    def _raise_news(ticker):
        raise Exception("news feed down")

    monkeypatch.setattr(cr, "fetch_company_news", _raise_news)
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)  # exercise the fallback path

    result = cr._handle_ticker_question("AAPL", "what's going on with this", [])

    assert "Buy" in result


def test_handle_ticker_question_falls_back_to_the_raw_summary_on_llm_outage(monkeypatch):
    insights = {"rating": "Hold", "fair_value_estimate": None, "current_price": None, "upside_percent": None, "flags": []}
    monkeypatch.setattr(cr, "build_stock_insights", lambda ticker: insights)
    monkeypatch.setattr(cr, "fetch_company_news", lambda ticker: [])
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)

    result = cr._handle_ticker_question("AAPL", "what's your source", [])

    assert "Hold" in result


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

    history = [{"role": "user", "content": "i just got 40000 as a student"}]
    result = cr._handle_advice_request("u1", "where should I invest", history)

    assert "aggressive" in result.lower()
    assert "Aggressive" in captured["prompt"]
    assert "Technology 100" in captured["prompt"]
    assert "500.00" in captured["prompt"]
    assert "i just got 40000 as a student" in captured["prompt"]


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

    result = cr._handle_advice_request("u1", "how should I spend $40k", [])

    assert "index funds" in result.lower()
    assert "hasn't completed the onboarding" in captured["prompt"]
    assert "No existing portfolio holdings" in captured["prompt"]


def test_handle_advice_request_falls_back_on_llm_outage(monkeypatch):
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: None)
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [])
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)

    result = cr._handle_advice_request("u1", "where should I invest", [])

    assert "allocation-level guidance" in result


# ---------------------------------------------------------------- _handle_stock_discovery_request

def test_handle_stock_discovery_request_is_a_fixed_consistent_message():
    # Deliberately not LLM-generated -- must return the exact same
    # answer every time, no monkeypatched provider involved at all.
    result = cr._handle_stock_discovery_request()
    assert "can't discover or recommend brand-new stocks" in result
    assert result == cr._handle_stock_discovery_request()  # identical on a second call


def test_classify_intent_routes_a_new_stock_request_to_discovery(monkeypatch):
    class _DiscoveryProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return "INTENT: stock_discovery_request"

    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _DiscoveryProvider)

    result = cr.classify_intent("find me two new companies to buy")

    assert result["intent"] == "stock_discovery_request"


def test_handle_chat_message_dispatches_stock_discovery_without_a_ticker(monkeypatch):
    class _DiscoveryProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return "INTENT: stock_discovery_request"

    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _DiscoveryProvider)

    result = cr.handle_chat_message("u1", "find me two new companies")

    assert result["intent"] == "stock_discovery_request"
    assert result["ticker"] is None
    assert "can't discover or recommend brand-new stocks" in result["reply"]


# ---------------------------------------------------------------- _handle_general

def test_handle_general_returns_the_llm_reply(monkeypatch):
    class _GeneralProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return "  I can help with that.  "

    monkeypatch.setattr(cr, "HostedProvider", _GeneralProvider)

    assert cr._handle_general("hello", []) == "I can help with that."


def test_handle_general_includes_recent_conversation_in_the_prompt(monkeypatch):
    captured = {}

    class _GeneralProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            captured["prompt"] = prompt
            return "reply"

    monkeypatch.setattr(cr, "HostedProvider", _GeneralProvider)

    history = [{"role": "assistant", "content": "Consider allocating $40k across index funds."}]
    cr._handle_general("what about that $40k thing", history)

    assert "Consider allocating $40k across index funds." in captured["prompt"]


def test_handle_general_falls_back_on_llm_outage(monkeypatch):
    monkeypatch.setattr(cr, "HostedProvider", _RaisingProvider)
    result = cr._handle_general("hello", [])
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


def test_handle_chat_message_defaults_history_to_empty(monkeypatch):
    # No history argument at all -- must not blow up, same as every
    # other optional-arg default in this module.
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _FakeProvider)
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"holdings": [], "summary": {}})

    result = cr.handle_chat_message("u1", "hows my portfolio")

    assert result["reply"]


# ---------------------------------------------------------------- _most_recent_ticker

def test_most_recent_ticker_finds_the_latest_ticker_walking_backward():
    history = [
        {"role": "user", "content": "what about Bajaj Finance"},
        {"role": "assistant", "content": "...", "ticker": "BAJFINANCE.NS"},
        {"role": "user", "content": "and Apple?"},
        {"role": "assistant", "content": "...", "ticker": "AAPL"},
    ]
    assert cr._most_recent_ticker(history) == "AAPL"


def test_most_recent_ticker_none_when_no_message_carries_one():
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi", "ticker": None}]
    assert cr._most_recent_ticker(history) is None


def test_most_recent_ticker_empty_history():
    assert cr._most_recent_ticker([]) is None


# ---------------------------------------------------------------- classify_intent (history-aware)

def test_classify_intent_carries_over_a_ticker_from_history_when_none_in_the_message(monkeypatch):
    class _FitProvider:
        def __init__(self, model=None, **kwargs):
            pass

        def generate(self, prompt, max_new_tokens=150):
            return "INTENT: portfolio_fit"

    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])  # nothing in THIS message
    monkeypatch.setattr(cr, "HostedProvider", _FitProvider)

    history = [
        {"role": "user", "content": "what about Bajaj Finance"},
        {"role": "assistant", "content": "...", "ticker": "BAJFINANCE.NS"},
    ]
    result = cr.classify_intent("does it fit my portfolio", history)

    assert result == {"intent": "portfolio_fit", "ticker": "BAJFINANCE.NS"}


def test_classify_intent_prefers_a_ticker_in_the_current_message_over_history(monkeypatch):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["MSFT"])
    monkeypatch.setattr(cr, "HostedProvider", _FakeProvider)

    history = [{"role": "assistant", "content": "...", "ticker": "AAPL"}]
    result = cr.classify_intent("what about Microsoft", history)

    assert result["ticker"] == "MSFT"


def test_classify_intent_history_defaults_to_empty(monkeypatch):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _FakeProvider)

    result = cr.classify_intent("hows my portfolio")

    assert result["ticker"] is None
