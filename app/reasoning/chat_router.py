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

Scope note: each message is classified independently, with no prior-
turn context fed into the classification prompt (so "what about
Bajaj Finance" then "does it fit my portfolio" won't resolve "it" to
the earlier ticker). GET /v1/chat/history already persists and
returns the full thread, so adding that context later is a prompt
change, not an architecture change -- not worth the added complexity
for this pass.
"""

import re

from app.api import db
from app.core.company_resolver import resolve_companies
from app.core.llm_provider import HostedProvider, LLMProviderError
from app.data.market_data import TickerNotFoundError
from app.reasoning.portfolio_fit import get_portfolio_fit_for_ticker
from app.reasoning.stock_score import build_stock_insights
from app.reporting.news_client import fetch_company_news
from app.reporting.portfolio_summary import build_portfolio_view

INTENT_PORTFOLIO_STATUS = "portfolio_status"
INTENT_TICKER_QUESTION = "ticker_question"
INTENT_PORTFOLIO_FIT = "portfolio_fit"
INTENT_FULL_REPORT_REQUEST = "full_report_request"
INTENT_GENERAL = "general"

_VALID_INTENTS = {
    INTENT_PORTFOLIO_STATUS, INTENT_TICKER_QUESTION, INTENT_PORTFOLIO_FIT,
    INTENT_FULL_REPORT_REQUEST, INTENT_GENERAL,
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


def _build_classify_prompt(message: str, has_ticker: bool) -> str:
    ticker_note = (
        "A ticker was already detected in this message by a separate, reliable lookup."
        if has_ticker else "No ticker was detected in this message."
    )
    return f"""Classify this message from a user of a stock research app into EXACTLY ONE of these categories:

portfolio_status - asking about their overall portfolio, holdings, or how it's doing
ticker_question - asking what's going on with, or for information about, a specific stock/company (not asking for a deep valuation report)
portfolio_fit - asking whether/how a specific stock fits, overlaps with, or diversifies their existing portfolio
full_report_request - explicitly asking for a full/deep/detailed research report, DCF, or valuation
general - anything else, including generic finance chit-chat

{ticker_note}

MESSAGE: {message}

Respond in EXACTLY this format, nothing else:
INTENT: <one of the categories above>"""


def _parse_intent(raw_text: str) -> str:
    match = _INTENT_RE.search(raw_text)
    label = match.group(1).lower() if match else ""
    return label if label in _VALID_INTENTS else INTENT_GENERAL


def classify_intent(message: str) -> dict:
    """{"intent": str, "ticker": str | None} -- ticker always comes
    from resolve_companies() (deterministic), never from the LLM.
    Falls back to INTENT_GENERAL on any classification failure -- a
    reply that's a bit too generic is a much smaller failure than a
    broken turn."""
    tickers = resolve_companies(message)
    ticker = tickers[0] if tickers else None

    try:
        provider = HostedProvider(model=_CHAT_MODEL)
        raw = provider.generate(_build_classify_prompt(message, ticker is not None), max_new_tokens=20)
        intent = _parse_intent(raw)
    except LLMProviderError:
        intent = INTENT_GENERAL

    if intent in _TICKER_SCOPED_INTENTS and not ticker:
        intent = INTENT_GENERAL

    return {"intent": intent, "ticker": ticker}


def _handle_portfolio_status(user_id: str) -> str:
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

    return "\n".join(lines)


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


def _handle_general(message: str) -> str:
    """Short, caveated LLM answer with no specific data context -- no
    portfolio/ticker to key off for this message, so this deliberately
    stays generic rather than fabricating specifics."""
    prompt = (
        "You are a terse assistant inside a stock research app. Answer the user's message in 1-3 sentences. "
        "You have no access to their portfolio or any specific stock data for this message -- if the question "
        "needs that, say so and suggest asking about a specific ticker or their portfolio instead. Never give "
        "personalized investment advice; you are not a licensed advisor.\n\n"
        f"MESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=_CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=150).strip()
    except LLMProviderError:
        return "I can help with your portfolio or a specific ticker -- try asking about one of those."


_HANDLERS = {
    INTENT_PORTFOLIO_STATUS: lambda user_id, ticker, message: _handle_portfolio_status(user_id),
    INTENT_TICKER_QUESTION: lambda user_id, ticker, message: _handle_ticker_question(ticker),
    INTENT_PORTFOLIO_FIT: lambda user_id, ticker, message: _handle_portfolio_fit(user_id, ticker),
    INTENT_FULL_REPORT_REQUEST: lambda user_id, ticker, message: _handle_full_report_request(ticker),
    INTENT_GENERAL: lambda user_id, ticker, message: _handle_general(message),
}


def handle_chat_message(user_id: str, message: str) -> dict:
    """{"reply": str, "intent": str, "ticker": str | None} -- never
    raises; every handler already degrades gracefully on its own, and
    classify_intent() itself falls back to INTENT_GENERAL on failure."""
    classification = classify_intent(message)
    intent = classification["intent"]
    ticker = classification["ticker"]

    reply = _HANDLERS[intent](user_id, ticker, message)

    return {"reply": reply, "intent": intent, "ticker": ticker}
