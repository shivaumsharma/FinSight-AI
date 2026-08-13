"""
chat_router.py

The fast conversational assistant's brain: classify a free-text
message into one of a small, fixed set of intents, then answer it
directly from already-computed data -- NOT the minutes-long job/poll
ResearchAgent pipeline (see app/api/jobs.py's own docstring:
MAX_CONCURRENT_JOBS defaults to 1, so that pipeline is fundamentally
serialized; chat must not inherit that bottleneck).

Intent classification is a single short HostedProvider.generate() call
over a small label set, parsed with the exact same regex-based
structured-output pattern app/reasoning/model_consensus.py's
_parse_opinion already established. resolve_companies() runs first as
a cheap, reliable pre-pass so the LLM only has to pick a label, never
has to spell a ticker out of free text itself -- same reasoning
company_resolver.py's own module docstring gives for deterministic
ticker resolution over an LLM guess.

Multi-turn context: handle_chat_message() takes an optional `history`
(the same list shape GET /v1/chat/history already returns/persists via
db.list_chat_messages -- oldest first). Both classify_intent() and the
handlers that synthesize an LLM answer (portfolio_status, advice_request,
general) fold the last few turns into their prompts, so "what about
Bajaj Finance" then "does it fit my portfolio" resolves "it" correctly.
Ticker carry-over is still deterministic, not an LLM guess: it reuses
the `ticker` a PRIOR assistant turn already resolved (stored by
add_chat_message's own `ticker` column), the same "never let the LLM
spell a ticker" reasoning company_resolver.py's docstring gives.
"""

import re

from app.api import db
from app.core.company_resolver import resolve_companies
from app.core.llm_provider import HostedProvider, LLMProviderError
from app.data.market_data import TickerNotFoundError
from app.reasoning.portfolio_fit import get_portfolio_fit_for_ticker, get_portfolio_sector_allocation
from app.reasoning.real_estate_guidance import get_real_estate_guidance
from app.reasoning.stock_score import build_stock_insights
from app.reporting.news_client import fetch_company_news
from app.reporting.portfolio_summary import build_portfolio_view

INTENT_PORTFOLIO_STATUS = "portfolio_status"
INTENT_TICKER_QUESTION = "ticker_question"
INTENT_PORTFOLIO_FIT = "portfolio_fit"
INTENT_FULL_REPORT_REQUEST = "full_report_request"
INTENT_ADVICE_REQUEST = "advice_request"
INTENT_GENERAL = "general"

_VALID_INTENTS = {
    INTENT_PORTFOLIO_STATUS, INTENT_TICKER_QUESTION, INTENT_PORTFOLIO_FIT,
    INTENT_FULL_REPORT_REQUEST, INTENT_ADVICE_REQUEST, INTENT_GENERAL,
}
# Intents that need a ticker to mean anything -- a message classified
# into one of these with no ticker detected degrades to INTENT_GENERAL
# in classify_intent() below, since there's nothing to look up.
_TICKER_SCOPED_INTENTS = {INTENT_TICKER_QUESTION, INTENT_PORTFOLIO_FIT, INTENT_FULL_REPORT_REQUEST}

_INTENT_RE = re.compile(r"INTENT:\s*(\w+)", re.IGNORECASE)

# A small, fast model is enough for a 5-way classification and a short
# generic reply -- same model_consensus.py precedent of picking a
# cheap model for a short, low-stakes call rather than the biggest
# available one.
_CHAT_MODEL = "llama-3.3-70b-versatile"

# How many recent messages (not turns -- individual user+assistant rows)
# to fold into a prompt as conversation memory. Small on purpose: this
# is a fast/cheap classification+synthesis call, not the full research
# pipeline, and a short window is enough to resolve "it"/"that" without
# growing the prompt (and the per-call cost/latency) unboundedly as a
# thread gets long.
MAX_HISTORY_MESSAGES = 6


def _format_history(history: list) -> str:
    if not history:
        return "(no prior messages in this conversation)"
    lines = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def _most_recent_ticker(history: list) -> "str | None":
    """Ticker carry-over for a follow-up with no ticker of its own
    (e.g. "does it fit my portfolio" after "what about Bajaj Finance").
    Walks backward through history for the most recent message that
    already carries a resolved `ticker` (only assistant turns do --
    see add_chat_message's `ticker` column) -- reuses a PRIOR
    deterministic resolve_companies() result rather than asking the
    LLM to guess a ticker out of old free text."""
    for msg in reversed(history[-MAX_HISTORY_MESSAGES:]):
        if msg.get("ticker"):
            return msg["ticker"]
    return None


def _build_classify_prompt(message: str, has_ticker: bool, history: list) -> str:
    ticker_note = (
        "A ticker is available for this message (detected directly, or carried over from recent "
        "conversation context below)."
        if has_ticker else "No ticker is available for this message."
    )
    return f"""Classify this message from a user of a stock research app into EXACTLY ONE of these categories:

portfolio_status - asking about their overall portfolio, holdings, or how it's doing
ticker_question - asking what's going on with, or for information about, a specific stock/company (not asking for a deep valuation report)
portfolio_fit - asking whether/how a specific stock fits, overlaps with, or diversifies their existing portfolio
full_report_request - explicitly asking for a full/deep/detailed research report, DCF, or valuation
advice_request - asking where/how to invest, allocate, or spend money based on their own goals or situation (e.g. "where should I invest", "how should I spend $40k", "what should a beginner do") -- NOT asking about an existing holding's status
general - anything else, including generic finance chit-chat

{ticker_note}

RECENT CONVERSATION (oldest first, context only -- classify the LATEST message, not this history):
{_format_history(history)}

LATEST MESSAGE: {message}

Respond in EXACTLY this format, nothing else:
INTENT: <one of the categories above>"""


def _parse_intent(raw_text: str) -> str:
    match = _INTENT_RE.search(raw_text)
    label = match.group(1).lower() if match else ""
    return label if label in _VALID_INTENTS else INTENT_GENERAL


def classify_intent(message: str, history: list = None) -> dict:
    """{"intent": str, "ticker": str | None} -- ticker comes from
    resolve_companies() on the CURRENT message first, falling back to
    _most_recent_ticker(history) for a ticker-less follow-up; never
    from the LLM. Falls back to INTENT_GENERAL on any classification
    failure -- a reply that's a bit too generic is a much smaller
    failure than a broken turn."""
    history = history or []
    tickers = resolve_companies(message)
    ticker = tickers[0] if tickers else _most_recent_ticker(history)

    try:
        provider = HostedProvider(model=_CHAT_MODEL)
        raw = provider.generate(_build_classify_prompt(message, ticker is not None, history), max_new_tokens=20)
        intent = _parse_intent(raw)
    except LLMProviderError:
        intent = INTENT_GENERAL

    if intent in _TICKER_SCOPED_INTENTS and not ticker:
        intent = INTENT_GENERAL

    return {"intent": intent, "ticker": ticker}


def _handle_portfolio_status(user_id: str, message: str, history: list) -> str:
    """Unlike the other handlers, this one's raw data assembly (lines,
    below) doubles as both the LLM's grounding context AND the
    LLM-outage fallback -- so a generic "how's my portfolio doing"
    still gets a real answer, but so does a specific "is this
    diversified enough" or "what's going well/badly", instead of the
    same fixed holdings dump regardless of what was actually asked."""
    view = build_portfolio_view(user_id)
    holdings = view["holdings"]
    if not holdings:
        return "You don't have any holdings in your portfolio yet."

    lines = [f"Your portfolio has {len(holdings)} holding{'s' if len(holdings) != 1 else ''}."]

    total_value = view["summary"]["total_market_value"]
    pnl = view["summary"]["total_unrealized_pnl"]
    pnl_pct = view["summary"]["total_unrealized_pnl_pct"]
    if total_value is not None:
        if pnl is not None:
            direction = "up" if pnl >= 0 else "down"
            lines.append(f"Total value: ${total_value:,.2f} (USD equiv.), {direction} ${abs(pnl):,.2f} ({pnl_pct:+.1f}%) overall.")
        else:
            lines.append(f"Total value: ${total_value:,.2f} (USD equiv.).")

    for h in holdings[:10]:
        price_bit = f"{h['price']:.2f} {h['currency']}" if h["price"] is not None else "price unavailable"
        rating_bit = f", rated {h['rating']}" if h.get("rating") else ""
        lines.append(f"- {h['ticker']}: {h['quantity']:g} sh @ {price_bit}{rating_bit}")

    allocation = get_portfolio_sector_allocation(db.get_portfolio_holdings(user_id))
    if allocation:
        top_sectors = sorted(allocation.items(), key=lambda kv: -kv[1])
        lines.append("Sector mix: " + ", ".join(f"{s} {p:g}%" for s, p in top_sectors) + ".")

    portfolio_data = "\n".join(lines)

    prompt = (
        "You are a terse assistant inside a stock research app, answering a question about the user's own "
        "portfolio. Use ONLY the real data below -- never invent numbers. Answer the user's actual question in "
        "2-4 sentences: if they ask about diversification, reference the sector mix; if they ask what's going "
        "well/badly, reference the per-holding ratings and overall P&L; if the question is generic, summarize "
        "total value and P&L. If the message references something from earlier in the conversation (e.g. "
        "'that', 'it'), use the recent conversation below to understand what they mean. Never give a buy/sell "
        "recommendation beyond restating existing ratings; never claim to be a licensed advisor.\n\n"
        f"PORTFOLIO DATA:\n{portfolio_data}\n\n"
        f"RECENT CONVERSATION:\n{_format_history(history)}\n\nMESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=_CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=200).strip()
    except LLMProviderError:
        return portfolio_data


def _handle_ticker_question(ticker: str) -> str:
    try:
        insights = build_stock_insights(ticker)
    except TickerNotFoundError:
        return f"I couldn't find {ticker}."

    lines = []
    if insights["rating"] and insights["rating"] != "Insufficient Data":
        lines.append(f"{ticker}: algorithmic rating is {insights['rating']}.")
        if insights["fair_value_estimate"] is not None and insights["current_price"] is not None:
            lines.append(
                f"Fair value estimate ${insights['fair_value_estimate']:,.2f} vs current price "
                f"${insights['current_price']:,.2f} ({insights['upside_percent']:+.1f}%)."
            )
        if insights["flags"]:
            lines.append("Flags: " + ", ".join(insights["flags"]) + ".")
    else:
        lines.append(f"{ticker}: not enough data for an algorithmic rating right now.")

    try:
        articles = fetch_company_news(ticker)
    except Exception:
        articles = []
    if articles:
        lines.append("Recent news: " + "; ".join(a["headline"] for a in articles[:3]) + ".")

    return "\n".join(lines)


def _handle_portfolio_fit(user_id: str, ticker: str) -> str:
    holdings = db.get_portfolio_holdings(user_id)
    return get_portfolio_fit_for_ticker(ticker, holdings)["summary"]


def _handle_full_report_request(ticker: str) -> str:
    return (
        f"That needs the full research pipeline for {ticker} (SEC filings, DCF valuation) -- "
        f"a couple of minutes, not seconds. Use \"Run Full Research Report\" to start it."
    )


def _handle_advice_request(user_id: str, message: str, history: list) -> str:
    """Grounded, portfolio-aware guidance -- pulls real holdings, sector
    allocation, and onboarding preferences (risk tolerance/goal/horizon)
    into the prompt so the answer isn't a blanket "I have no access"
    disclaimer. Stays at the same allocation-level, non-stock-picking
    register as real_estate_guidance.py/portfolio_fit.py rather than
    naming a specific buy/sell call -- informational, not licensed
    advice (same boundary _handle_general already states, just backed
    by real numbers here instead of nothing)."""
    user = db.get_user_by_id(user_id)
    holdings = db.get_portfolio_holdings(user_id)

    context_lines = []
    if user and user.get("onboarding_completed"):
        context_lines.append(
            f"Risk tolerance: {user.get('risk_tolerance') or 'unset'}. "
            f"Goal: {user.get('investment_goal') or 'unset'}. "
            f"Horizon: {user.get('investment_horizon') or 'unset'}."
        )
    else:
        context_lines.append("This user hasn't completed the onboarding preferences questionnaire yet.")

    if holdings:
        view = build_portfolio_view(user_id)
        total_value = view["summary"].get("total_market_value")
        allocation = get_portfolio_sector_allocation(holdings)
        top_sectors = sorted(allocation.items(), key=lambda kv: -kv[1])[:5]
        sector_bit = ", ".join(f"{s} {p:g}%" for s, p in top_sectors)
        value_bit = f", ${total_value:,.2f} total (USD equiv.)" if total_value is not None else ""
        context_lines.append(
            f"Current portfolio: {len(holdings)} holding(s){value_bit}."
            + (f" Sector mix: {sector_bit}." if sector_bit else "")
        )
    else:
        context_lines.append("No existing portfolio holdings.")

    if user and user.get("interested_in_real_estate"):
        re_guidance = get_real_estate_guidance(user)
        if re_guidance:
            context_lines.append(
                f"Opted into real estate interest -- target allocation "
                f"{re_guidance['target_allocation_low_pct']:g}-{re_guidance['target_allocation_high_pct']:g}% "
                f"per their risk tolerance."
            )

    prompt = (
        "You are a terse assistant inside a stock research app, answering a request for investing/allocation "
        "guidance. Use ONLY the real context below -- never invent numbers. Answer in 2-4 sentences, at the "
        "level of asset classes/sectors/allocation percentages, not a specific buy/sell call on an individual "
        "stock. If onboarding preferences are unset, give general starter guidance (e.g. diversified index "
        "funds first) but mention completing onboarding for guidance tailored to their risk tolerance. If the "
        "message references something from earlier in the conversation (e.g. 'that', 'the $40k thing'), use "
        "the recent conversation below to understand what they mean. Never claim to be a licensed advisor.\n\n"
        "CONTEXT:\n" + "\n".join(context_lines)
        + f"\n\nRECENT CONVERSATION:\n{_format_history(history)}\n\nMESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=_CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=200).strip()
    except LLMProviderError:
        return "I can give allocation-level guidance using your real portfolio and preferences, but couldn't generate a response just now -- try again in a moment."


def _handle_general(message: str, history: list) -> str:
    """Short, caveated LLM answer with no specific data context -- no
    portfolio/ticker to key off for this message, so this deliberately
    stays generic rather than fabricating specifics. Still gets the
    recent conversation, purely so a reference to an earlier turn reads
    as understood rather than ignored -- it has no new data to answer
    with regardless."""
    prompt = (
        "You are a terse assistant inside a stock research app. Answer the user's message in 1-3 sentences. "
        "You have no access to their portfolio or any specific stock data for this message -- if the question "
        "needs that, say so and suggest asking about a specific ticker or their portfolio instead. Use the "
        "recent conversation below only to understand references like 'that' or 'it' -- never give "
        "personalized investment advice; you are not a licensed advisor.\n\n"
        f"RECENT CONVERSATION:\n{_format_history(history)}\n\nMESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=_CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=150).strip()
    except LLMProviderError:
        return "I can help with your portfolio or a specific ticker -- try asking about one of those."


_HANDLERS = {
    INTENT_PORTFOLIO_STATUS: lambda user_id, ticker, message, history: _handle_portfolio_status(user_id, message, history),
    INTENT_TICKER_QUESTION: lambda user_id, ticker, message, history: _handle_ticker_question(ticker),
    INTENT_PORTFOLIO_FIT: lambda user_id, ticker, message, history: _handle_portfolio_fit(user_id, ticker),
    INTENT_FULL_REPORT_REQUEST: lambda user_id, ticker, message, history: _handle_full_report_request(ticker),
    INTENT_ADVICE_REQUEST: lambda user_id, ticker, message, history: _handle_advice_request(user_id, message, history),
    INTENT_GENERAL: lambda user_id, ticker, message, history: _handle_general(message, history),
}


def handle_chat_message(user_id: str, message: str, history: list = None) -> dict:
    """{"reply": str, "intent": str, "ticker": str | None} -- never
    raises; every handler already degrades gracefully on its own, and
    classify_intent() itself falls back to INTENT_GENERAL on failure.

    `history` is GET /v1/chat/history's own list shape (oldest first,
    from db.list_chat_messages) -- the caller's job to fetch it BEFORE
    persisting the current message, so it never includes the message
    being classified right now."""
    history = history or []
    classification = classify_intent(message, history)
    intent = classification["intent"]
    ticker = classification["ticker"]

    reply = _HANDLERS[intent](user_id, ticker, message, history)

    return {"reply": reply, "intent": intent, "ticker": ticker}
