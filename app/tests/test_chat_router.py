"""
Unit tests for app/reasoning/chat_router.py. HostedProvider is
monkeypatched to a fake class per test (same convention as
test_model_consensus.py) -- never hits real network.
"""

import pytest

from app.api import db
from app.core.llm_provider import LLMProviderError
from app.data.market_data import TickerNotFoundError
from app.reasoning import chat_router as cr


@pytest.fixture(autouse=True)
def _clear_pending_orders():
    # _pending_orders is module-scoped, per-user, in-memory state (see
    # its own comment in chat_router.py) -- give each test a clean slate
    # so one test's unconfirmed order can't leak into another's.
    cr._pending_orders.clear()
    yield
    cr._pending_orders.clear()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    # Real sqlite, not mocked -- db.execute_order/get_portfolio_holdings
    # do real transactional work the order-placement tests below need to
    # actually exercise, same reasoning test_orders.py's own temp_db
    # fixture gives.
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


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


# ---------------------------------------------------------------- _parse_side / _parse_quantity / _parse_confirmation

def test_parse_side_detects_buy():
    assert cr._parse_side("buy 5 AAPL") == "BUY"


def test_parse_side_detects_sell():
    assert cr._parse_side("sell 10 TCS") == "SELL"


def test_parse_side_is_case_insensitive():
    assert cr._parse_side("BUY 5 AAPL") == "BUY"


def test_parse_side_prefers_whichever_keyword_appears_first():
    assert cr._parse_side("I already sold today, buy 5 more AAPL") == "BUY"


def test_parse_side_none_when_neither_keyword_present():
    assert cr._parse_side("what about AAPL") is None


def test_parse_quantity_extracts_the_first_number():
    assert cr._parse_quantity("sell 10 TCS") == 10.0


def test_parse_quantity_handles_decimals():
    assert cr._parse_quantity("buy 2.5 shares of AAPL") == 2.5


def test_parse_quantity_none_when_no_number_present():
    assert cr._parse_quantity("buy some AAPL") is None


def test_parse_quantity_none_for_zero():
    assert cr._parse_quantity("buy 0 AAPL") is None


def test_parse_confirmation_recognizes_yes_variants():
    for msg in ["yes", "Yes", "yeah", "confirm", "do it", "go ahead", "yes, place it"]:
        assert cr._parse_confirmation(msg) == "yes", msg


def test_parse_confirmation_recognizes_no_variants():
    for msg in ["no", "No", "cancel", "nevermind", "no, don't"]:
        assert cr._parse_confirmation(msg) == "no", msg


def test_parse_confirmation_none_for_an_ambiguous_reply():
    assert cr._parse_confirmation("what's the price of AAPL") is None


def test_parse_confirmation_does_not_match_a_word_that_merely_contains_no():
    # "know" contains the letters "n"/"o" but is not the word "no" --
    # must not be read as a "no" to a pending order.
    assert cr._parse_confirmation("you know, TCS had a strong quarter") is None


# ---------------------------------------------------------------- _handle_place_order (turn 1: propose)

def test_handle_place_order_with_no_ticker():
    result = cr._handle_place_order("u1", None, "buy 5 shares", [])
    assert "couldn't tell which stock" in result


def test_handle_place_order_with_no_side(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [])
    result = cr._handle_place_order("u1", "AAPL", "what about 5 AAPL", [])
    assert "buy or a sell" in result


def test_handle_place_order_sell_with_no_quantity_asks_without_fetching_a_quote(monkeypatch):
    # SELL never gets a sizing suggestion (Feature 4 only applies to
    # BUY) -- must reject before ever calling get_quote.
    def _unexpected_quote(ticker):
        raise AssertionError("get_quote should not be called for a quantity-less SELL")

    monkeypatch.setattr(cr, "get_quote", _unexpected_quote)
    result = cr._handle_place_order("u1", "TCS", "sell some TCS", [])
    assert "How many shares of TCS" in result
    assert "u1" not in cr._pending_orders


def test_handle_place_order_buy_with_no_quantity_and_no_portfolio_value_asks_for_a_quantity(monkeypatch):
    # Feature 4's suggestion has nothing real to size off of here -- an
    # honest "how many" fallback, never a fabricated default quantity.
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {"risk_tolerance": "Moderate"})
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"summary": {"total_market_value": None}})

    result = cr._handle_place_order("u1", "AAPL", "buy some AAPL", [])

    assert "How many shares of AAPL" in result
    assert "u1" not in cr._pending_orders


def test_handle_place_order_sell_exceeding_holdings(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [{"ticker": "TCS", "quantity": 5}])
    result = cr._handle_place_order("u1", "TCS", "sell 1000 TCS", [])
    assert "only hold 5" in result
    assert "u1" not in cr._pending_orders


def test_handle_place_order_sell_with_no_holding_at_all(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [])
    result = cr._handle_place_order("u1", "TCS", "sell 10 TCS", [])
    assert "don't currently hold any TCS" in result


def test_handle_place_order_degrades_honestly_on_a_quote_failure(monkeypatch):
    monkeypatch.setattr(cr, "get_quote", lambda ticker: (_ for _ in ()).throw(Exception("yfinance down")))
    result = cr._handle_place_order("u1", "AAPL", "buy 5 AAPL", [])
    assert "couldn't check AAPL's price" in result
    assert "u1" not in cr._pending_orders


def test_handle_place_order_buy_happy_path_stores_a_pending_order_and_asks_to_confirm(monkeypatch):
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})
    result = cr._handle_place_order("u1", "AAPL", "buy 5 AAPL", [])

    assert result == 'Confirm: BUY 5 AAPL at today\'s price of $150.00? Reply "yes" to place the order or "no" to cancel.'
    assert cr._pending_orders["u1"]["legs"] == [
        {"ticker": "AAPL", "side": "BUY", "quantity": 5.0, "price": 150.0, "currency": "USD", "rationale": "User-initiated via chat"}
    ]


def test_handle_place_order_sell_within_holdings_uses_the_rupee_symbol(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [{"ticker": "TCS", "quantity": 10}])
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 4231.0, "currency": "INR"})
    result = cr._handle_place_order("u1", "TCS", "sell 10 TCS", [])
    assert "₹4,231.00" in result


# ---------------------------------------------------------------- Feature 4: position sizing

def testsuggest_quantity_scales_with_risk_tolerance(monkeypatch):
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {"risk_tolerance": "Aggressive"})
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"summary": {"total_market_value": 10_000.0}})

    quantity, note = cr.suggest_quantity("u1", "AAPL", 100.0)

    assert quantity == 10.0  # 10% of $10,000 / $100
    assert "10%" in note
    assert "Aggressive" in note


def testsuggest_quantity_defaults_to_moderate_when_unset(monkeypatch):
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {"risk_tolerance": None})
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"summary": {"total_market_value": 10_000.0}})

    quantity, note = cr.suggest_quantity("u1", "AAPL", 100.0)

    assert quantity == 5.0  # 5% of $10,000 / $100
    assert "Moderate" in note


def testsuggest_quantity_none_for_an_empty_portfolio(monkeypatch):
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {"risk_tolerance": "Moderate"})
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"summary": {"total_market_value": None}})

    assert cr.suggest_quantity("u1", "AAPL", 100.0) is None


def testsuggest_quantity_two_users_same_value_different_risk_tolerance_diverge(monkeypatch):
    # The spec's own acceptance check: identical portfolio value, two
    # different risk_tolerance settings -- Aggressive suggests more.
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"summary": {"total_market_value": 20_000.0}})

    users = {"conservative_user": "Conservative", "aggressive_user": "Aggressive"}
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {"risk_tolerance": users[user_id]})

    conservative_qty, _ = cr.suggest_quantity("conservative_user", "AAPL", 100.0)
    aggressive_qty, _ = cr.suggest_quantity("aggressive_user", "AAPL", 100.0)

    assert aggressive_qty > conservative_qty


def test_handle_place_order_buy_with_no_quantity_proposes_a_sized_suggestion(monkeypatch):
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda user_id: {"risk_tolerance": "Conservative"})
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"summary": {"total_market_value": 15_000.0}})

    result = cr._handle_place_order("u1", "AAPL", "buy AAPL", [])

    assert "Suggest 2 shares of AAPL" in result  # 2% of $15,000 / $150
    assert "2%" in result
    assert "Conservative" in result
    assert "confirm, or tell me a different quantity" in result
    leg = cr._pending_orders["u1"]["legs"][0]
    assert leg == {
        "ticker": "AAPL", "side": "BUY", "quantity": 2.0, "price": 150.0, "currency": "USD",
        "rationale": "User-initiated via chat, sized via suggestion (that's about 2% of your portfolio, sized for Conservative)",
    }


def test_handle_chat_message_confirming_a_sized_suggestion_executes_it(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["AAPL"])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: place_order",
    }))
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 100.0, "currency": "USD"})
    monkeypatch.setattr(cr.db, "get_user_by_id", lambda uid: {"risk_tolerance": "Aggressive"})
    monkeypatch.setattr(cr, "build_portfolio_view", lambda uid: {"summary": {"total_market_value": 5_000.0}})

    propose = cr.handle_chat_message(user_id, "buy AAPL")
    assert "Suggest 5 shares of AAPL" in propose["reply"]  # 10% of $5,000 / $100

    confirm = cr.handle_chat_message(user_id, "yes")
    assert "Done -- BUY 5 AAPL" in confirm["reply"]

    orders = temp_db.list_orders(user_id)
    assert orders[0]["quantity"] == 5.0
    assert "sized via suggestion" in orders[0]["rationale"]


# ---------------------------------------------------------------- handle_chat_message (full two-turn flow)

def test_handle_chat_message_confirm_yes_executes_the_order(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["AAPL"])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: place_order",
    }))
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 150.0, "currency": "USD"})

    propose = cr.handle_chat_message(user_id, "buy 4 AAPL")
    assert propose["intent"] == "place_order"
    assert "Confirm: BUY 4 AAPL" in propose["reply"]

    confirm = cr.handle_chat_message(user_id, "yes")
    assert "Done -- BUY 4 AAPL filled at $150.00" in confirm["reply"]
    assert confirm["ticker"] == "AAPL"

    orders = temp_db.list_orders(user_id)
    assert len(orders) == 1
    assert orders[0]["ticker"] == "AAPL"
    assert orders[0]["quantity"] == 4.0
    assert orders[0]["rationale"] == "User-initiated via chat"
    holdings = {h["ticker"]: h for h in temp_db.get_portfolio_holdings(user_id)}
    assert holdings["AAPL"]["quantity"] == 4.0

    # Confirmed and cleared -- a second "yes" has nothing left to confirm.
    assert user_id not in cr._pending_orders


def test_handle_chat_message_confirm_no_cancels_without_placing_an_order(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    cr._pending_orders[user_id] = {
        "legs": [{"ticker": "AAPL", "side": "BUY", "quantity": 5.0, "price": 150.0, "currency": "USD", "rationale": None}],
    }

    result = cr.handle_chat_message(user_id, "no")

    assert result["reply"] == "Cancelled -- no order placed."
    assert user_id not in cr._pending_orders
    assert temp_db.list_orders(user_id) == []


def test_handle_chat_message_ambiguous_reply_drops_the_pending_order_and_classifies_normally(monkeypatch):
    cr._pending_orders["u1"] = {
        "legs": [{"ticker": "AAPL", "side": "BUY", "quantity": 5.0, "price": 150.0, "currency": "USD", "rationale": None}],
    }
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", _FakeProvider)  # returns "INTENT: portfolio_status"
    monkeypatch.setattr(cr, "build_portfolio_view", lambda user_id: {"holdings": [], "summary": {}})

    result = cr.handle_chat_message("u1", "actually hows my portfolio doing")

    assert "u1" not in cr._pending_orders
    assert result["intent"] == "portfolio_status"


def test_handle_chat_message_sell_more_than_held_is_a_clear_rejection_not_a_crash(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "TCS", "BUY", 5, 4000.0, "INR")

    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["TCS"])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: place_order",
    }))

    result = cr.handle_chat_message(user_id, "sell 1000 TCS")

    assert "only hold 5" in result["reply"]
    assert user_id not in cr._pending_orders  # never proposed, nothing to confirm
    # Only the setup BUY exists -- the rejected SELL never got recorded.
    orders = temp_db.list_orders(user_id)
    assert len(orders) == 1
    assert orders[0]["side"] == "BUY"


# ---------------------------------------------------------------- Feature 1: batch commands

def test_parse_batch_command_recognizes_sell_everything_rated():
    assert cr._parse_batch_command("sell everything rated Sell") == {"type": "sell_rated", "rating": "Sell"}


def test_parse_batch_command_sell_rated_is_case_insensitive():
    assert cr._parse_batch_command("Sell Everything Rated hold") == {"type": "sell_rated", "rating": "Hold"}


def test_parse_batch_command_recognizes_put_amount_into_top_n():
    result = cr._parse_batch_command("put $5000 into my top 3 Buy-rated watchlist stocks")
    assert result == {"type": "buy_top_n", "amount": 5000.0, "n": 3}


def test_parse_batch_command_recognizes_buy_across_top_n_with_amount():
    result = cr._parse_batch_command("buy across my top 3 buy-rated watchlist stocks with $5000")
    assert result == {"type": "buy_top_n", "n": 3, "amount": 5000.0}


def test_parse_batch_command_handles_comma_separated_amounts():
    result = cr._parse_batch_command("put ₹50,000 into my top 2 Buy-rated watchlist stocks")
    assert result == {"type": "buy_top_n", "amount": 50000.0, "n": 2}


def test_parse_batch_command_none_for_a_single_ticker_order():
    assert cr._parse_batch_command("buy 5 AAPL") is None


def test_parse_batch_command_none_for_unrelated_text():
    assert cr._parse_batch_command("what's my portfolio worth") is None


def test_expand_sell_rated_batch_only_includes_matching_rating(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [
        {"ticker": "TCS", "quantity": 10}, {"ticker": "INFY", "quantity": 5}, {"ticker": "WIPRO", "quantity": 2},
    ])
    ratings = {"TCS": "Sell", "INFY": "Sell", "WIPRO": "Hold"}
    monkeypatch.setattr(cr.db, "get_latest_rating_for_ticker", lambda user_id, ticker: ratings[ticker])
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 100.0, "currency": "INR"})

    legs, notes = cr._expand_sell_rated_batch("u1", "Sell")

    assert {l["ticker"] for l in legs} == {"TCS", "INFY"}
    assert all(l["side"] == "SELL" for l in legs)
    assert notes == []


def test_expand_sell_rated_batch_skips_and_notes_a_quote_failure(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [{"ticker": "TCS", "quantity": 10}])
    monkeypatch.setattr(cr.db, "get_latest_rating_for_ticker", lambda user_id, ticker: "Sell")

    def _raise(ticker):
        raise Exception("down")

    monkeypatch.setattr(cr, "get_quote", _raise)

    legs, notes = cr._expand_sell_rated_batch("u1", "Sell")

    assert legs == []
    assert "TCS" in notes[0]


def test_expand_buy_top_n_batch_splits_the_amount_evenly(monkeypatch):
    monkeypatch.setattr(cr.db, "get_watchlist", lambda user_id: [{"ticker": "AAPL"}, {"ticker": "MSFT"}, {"ticker": "TSLA"}])
    ratings = {"AAPL": "Buy", "MSFT": "Buy", "TSLA": "Hold"}
    monkeypatch.setattr(cr.db, "get_latest_rating_for_ticker", lambda user_id, ticker: ratings[ticker])
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 100.0, "currency": "USD"})

    legs, notes = cr._expand_buy_top_n_batch("u1", 3, 1000.0)

    assert {l["ticker"] for l in legs} == {"AAPL", "MSFT"}  # TSLA excluded -- not Buy-rated
    assert all(l["quantity"] == 5.0 for l in legs)  # $500 / $100 each
    assert "only 2 Buy-rated watchlist stock" in notes[0]


def test_expand_buy_top_n_batch_respects_watchlist_order_and_caps_at_n(monkeypatch):
    monkeypatch.setattr(cr.db, "get_watchlist", lambda user_id: [
        {"ticker": "AAPL"}, {"ticker": "MSFT"}, {"ticker": "TSLA"}, {"ticker": "NVDA"},
    ])
    monkeypatch.setattr(cr.db, "get_latest_rating_for_ticker", lambda user_id, ticker: "Buy")
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 100.0, "currency": "USD"})

    legs, notes = cr._expand_buy_top_n_batch("u1", 2, 1000.0)

    assert [l["ticker"] for l in legs] == ["AAPL", "MSFT"]  # first 2, watchlist order
    assert notes == []


def test_expand_buy_top_n_batch_empty_when_nothing_is_buy_rated(monkeypatch):
    monkeypatch.setattr(cr.db, "get_watchlist", lambda user_id: [{"ticker": "AAPL"}])
    monkeypatch.setattr(cr.db, "get_latest_rating_for_ticker", lambda user_id, ticker: "Hold")

    legs, notes = cr._expand_buy_top_n_batch("u1", 3, 1000.0)

    assert legs == []


def test_propose_batch_order_with_no_matches_reports_the_empty_case_without_a_pending_order(monkeypatch):
    monkeypatch.setattr(cr.db, "get_portfolio_holdings", lambda user_id: [])
    result = cr._propose_batch_order("u1", {"type": "sell_rated", "rating": "Sell"})
    assert "don't currently hold anything rated Sell" in result
    assert "u1" not in cr._pending_orders


def test_handle_chat_message_batch_sell_rated_end_to_end(temp_db, monkeypatch):
    # The spec's own acceptance scenario: a portfolio with 2 Sell-rated
    # and 1 Hold-rated holding, "sell everything rated Sell" -> the
    # confirmation shows exactly those 2 tickers, and only those 2
    # orders appear after confirming.
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.execute_order(user_id, "TCS", "BUY", 10, 4000.0, "INR")
    temp_db.execute_order(user_id, "INFY", "BUY", 5, 1500.0, "INR")
    temp_db.execute_order(user_id, "WIPRO", "BUY", 2, 400.0, "INR")

    ratings = {"TCS": "Sell", "INFY": "Sell", "WIPRO": "Hold"}
    monkeypatch.setattr(cr.db, "get_latest_rating_for_ticker", lambda user_id, ticker: ratings[ticker])
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 4200.0, "currency": "INR"})
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: place_order",
    }))

    propose = cr.handle_chat_message(user_id, "sell everything rated Sell")
    assert "This will sell:" in propose["reply"]
    assert "10 TCS" in propose["reply"]
    assert "5 INFY" in propose["reply"]
    assert "WIPRO" not in propose["reply"]

    confirm = cr.handle_chat_message(user_id, "yes")
    assert "Filled:" in confirm["reply"]

    orders = temp_db.list_orders(user_id)
    sell_orders = [o for o in orders if o["side"] == "SELL"]
    assert {o["ticker"] for o in sell_orders} == {"TCS", "INFY"}
    assert all(o["rationale"] == "Batch: sell everything rated Sell" for o in sell_orders)


# ---------------------------------------------------------------- Feature 3: price alerts

def test_parse_alert_command_recognizes_below_with_dollar_sign():
    result = cr._parse_alert_command("sell if AAPL drops below $180")
    assert result == {"direction": "below", "target_price": 180.0, "side": "SELL", "auto_execute": False}


def test_parse_alert_command_recognizes_above():
    result = cr._parse_alert_command("buy if AAPL rises above 200")
    assert result == {"direction": "above", "target_price": 200.0, "side": "BUY", "auto_execute": False}


def test_parse_alert_command_defaults_side_to_sell_with_no_side_word():
    result = cr._parse_alert_command("alert me if AAPL drops below 180")
    assert result == {"direction": "below", "target_price": 180.0, "side": "SELL", "auto_execute": False}


def test_parse_alert_command_auto_execute_requires_both_side_and_opt_in_word():
    # Opt-in word present but no side -- must NOT auto-execute (this
    # feature's own explicit guardrail: never the default from one
    # ambiguous command).
    result = cr._parse_alert_command("automatically alert me if AAPL drops below 180")
    assert result["auto_execute"] is False


def test_parse_alert_command_auto_execute_true_with_both_present():
    result = cr._parse_alert_command("automatically sell if AAPL drops below 180")
    assert result == {"direction": "below", "target_price": 180.0, "side": "SELL", "auto_execute": True}


def test_parse_alert_command_recognizes_auto_execute_hyphenated():
    result = cr._parse_alert_command("auto-execute: sell if AAPL drops below 180")
    assert result["auto_execute"] is True


def test_parse_alert_command_none_for_a_plain_order():
    assert cr._parse_alert_command("sell 10 TCS") is None


def test_handle_create_alert_with_no_ticker():
    result = cr._handle_create_alert("u1", None, "alert me if it drops below 180", [])
    assert "couldn't tell which stock" in result


def test_handle_create_alert_with_no_price_condition(monkeypatch):
    result = cr._handle_create_alert("u1", "AAPL", "watch AAPL for me", [])
    assert "couldn't tell what price condition" in result


def test_handle_create_alert_happy_path_persists_and_confirms(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 190.0, "currency": "USD"})

    result = cr._handle_create_alert(user_id, "AAPL", "sell if AAPL drops below $180", [])

    assert result == 'Alert set: if AAPL drops below $180.00, I\'ll let you know (simulated).'
    alerts = temp_db.list_price_alerts(user_id)
    assert len(alerts) == 1
    assert alerts[0] == {
        "alert_id": alerts[0]["alert_id"], "ticker": "AAPL", "side": "SELL", "direction": "below",
        "target_price": 180.0, "auto_execute": 0, "created_at": alerts[0]["created_at"], "triggered_at": None,
    }


def test_handle_create_alert_auto_execute_wording(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 4200.0, "currency": "INR"})

    result = cr._handle_create_alert(user_id, "TCS", "automatically sell if TCS drops below 4000", [])

    assert "automatically sell" in result
    assert "₹4,000.00" in result
    assert temp_db.list_price_alerts(user_id)[0]["auto_execute"] == 1


def test_handle_create_alert_degrades_currency_symbol_on_quote_failure(monkeypatch):
    monkeypatch.setattr(cr, "get_quote", lambda ticker: (_ for _ in ()).throw(Exception("down")))
    monkeypatch.setattr(cr.db, "create_price_alert", lambda *a, **kw: "alert-1")

    result = cr._handle_create_alert("u1", "AAPL", "sell if AAPL drops below 180", [])

    assert "$180.00" in result  # still confirms, just with the USD fallback symbol


def test_handle_watchlist_add_with_no_ticker():
    result = cr._handle_watchlist_add("u1", None)
    assert "couldn't tell which stock" in result


def test_handle_watchlist_add_happy_path_persists_and_confirms(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 190.0, "currency": "USD"})

    result = cr._handle_watchlist_add(user_id, "AAPL")

    assert result == "Added AAPL to your watchlist -- trading at $190.00 right now."
    assert {item["ticker"] for item in temp_db.get_watchlist(user_id)} == {"AAPL"}


def test_handle_watchlist_add_already_present(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "AAPL")
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 190.0, "currency": "USD"})

    result = cr._handle_watchlist_add(user_id, "AAPL")

    assert result == "AAPL is already on your watchlist."


def test_handle_watchlist_add_degrades_on_quote_failure(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "get_quote", lambda ticker: (_ for _ in ()).throw(Exception("down")))

    result = cr._handle_watchlist_add(user_id, "AAPL")

    assert result == "Added AAPL to your watchlist."  # still confirms, just without a price
    assert {item["ticker"] for item in temp_db.get_watchlist(user_id)} == {"AAPL"}


def test_handle_watchlist_remove_with_no_ticker():
    result = cr._handle_watchlist_remove("u1", None)
    assert "couldn't tell which stock" in result


def test_handle_watchlist_remove_happy_path(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "AAPL")

    result = cr._handle_watchlist_remove(user_id, "AAPL")

    assert result == "Removed AAPL from your watchlist."
    assert temp_db.get_watchlist(user_id) == []


def test_handle_watchlist_remove_not_present(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")

    result = cr._handle_watchlist_remove(user_id, "AAPL")

    assert result == "AAPL wasn't on your watchlist."


def test_handle_chat_message_routes_watchlist_add_intent(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["BAJFINANCE.NS"])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: watchlist_add",
    }))
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 4200.0, "currency": "INR"})

    result = cr.handle_chat_message(user_id, "add Bajaj Finance to my watchlist")

    assert result["intent"] == "watchlist_add"
    assert "Added BAJFINANCE.NS to your watchlist" in result["reply"]
    assert {item["ticker"] for item in temp_db.get_watchlist(user_id)} == {"BAJFINANCE.NS"}


def test_handle_chat_message_routes_watchlist_remove_intent(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "AAPL")
    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["AAPL"])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: watchlist_remove",
    }))

    result = cr.handle_chat_message(user_id, "remove AAPL from my watchlist")

    assert result["intent"] == "watchlist_remove"
    assert result["reply"] == "Removed AAPL from your watchlist."
    assert temp_db.get_watchlist(user_id) == []


def test_handle_list_alerts_with_none(temp_db):
    assert "don't have any active price alerts" in cr._handle_list_alerts("u1")


def test_handle_list_alerts_lists_active_alerts(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.create_price_alert(user_id, "AAPL", "SELL", "below", 180.0)
    temp_db.create_price_alert(user_id, "TCS", "BUY", "above", 4500.0, auto_execute=True)

    result = cr._handle_list_alerts(user_id)

    assert "AAPL drops below 180.00 -> SELL" in result
    assert "TCS rises above 4,500.00 -> BUY (auto)" in result


def test_handle_chat_message_routes_create_alert_intent(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    monkeypatch.setattr(cr, "resolve_companies", lambda q: ["AAPL"])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: create_alert",
    }))
    monkeypatch.setattr(cr, "get_quote", lambda ticker: {"price": 190.0, "currency": "USD"})

    result = cr.handle_chat_message(user_id, "alert me if AAPL drops below 180")

    assert result["intent"] == "create_alert"
    assert "Alert set" in result["reply"]
    assert len(temp_db.list_price_alerts(user_id)) == 1


def test_handle_chat_message_routes_list_alerts_intent_with_no_ticker_needed(temp_db, monkeypatch):
    monkeypatch.setattr(cr, "resolve_companies", lambda q: [])
    monkeypatch.setattr(cr, "HostedProvider", type("P", (), {
        "__init__": lambda self, model=None, **kw: None,
        "generate": lambda self, prompt, max_new_tokens=150: "INTENT: list_alerts",
    }))

    result = cr.handle_chat_message("u1", "what alerts do I have")

    assert result["intent"] == "list_alerts"
    assert "don't have any active price alerts" in result["reply"]


def test_execute_pending_order_reports_a_failed_leg_by_message(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    # Stale price a since-changed holding can no longer support -- the
    # leg was valid when proposed, but execute_order re-validates for
    # real at execution time and this must surface as a clear failure,
    # not a crash or a silent no-op.
    cr._pending_orders[user_id] = {
        "legs": [{"ticker": "TCS", "side": "SELL", "quantity": 10.0, "price": 4000.0, "currency": "INR", "rationale": None}],
    }

    result = cr._execute_pending_order(user_id)

    assert "Couldn't place that order" in result
    assert user_id not in cr._pending_orders
