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
import time

from app.api import db
from app.core.company_resolver import has_unresolved_ticker_attempt, resolve_companies
from app.core.currency import currency_symbol
from app.core.llm_provider import HostedProvider, LLMProviderError
from app.data.market_data import TickerNotFoundError, get_quote
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
INTENT_STOCK_DISCOVERY_REQUEST = "stock_discovery_request"
INTENT_PLACE_ORDER = "place_order"
INTENT_CREATE_ALERT = "create_alert"
INTENT_LIST_ALERTS = "list_alerts"
INTENT_WATCHLIST_ADD = "watchlist_add"
INTENT_WATCHLIST_REMOVE = "watchlist_remove"
INTENT_ADD_HOLDING = "add_holding"
INTENT_END_VOICE_SESSION = "end_voice_session"
INTENT_DAILY_BRIEFING = "daily_briefing"
INTENT_GENERAL = "general"

_VALID_INTENTS = {
    INTENT_PORTFOLIO_STATUS, INTENT_TICKER_QUESTION, INTENT_PORTFOLIO_FIT,
    INTENT_FULL_REPORT_REQUEST, INTENT_ADVICE_REQUEST, INTENT_STOCK_DISCOVERY_REQUEST,
    INTENT_PLACE_ORDER, INTENT_CREATE_ALERT, INTENT_LIST_ALERTS,
    INTENT_WATCHLIST_ADD, INTENT_WATCHLIST_REMOVE, INTENT_ADD_HOLDING,
    INTENT_END_VOICE_SESSION, INTENT_DAILY_BRIEFING, INTENT_GENERAL,
}
# Intents that need a ticker to mean anything -- a message classified
# into one of these with no ticker detected degrades to INTENT_GENERAL
# in classify_intent() below, since there's nothing to look up.
# INTENT_PLACE_ORDER is deliberately NOT here: a batch command ("sell
# everything rated Sell") is a perfectly valid order intent with no
# single ticker at all -- _handle_place_order below is the one that
# decides whether a missing ticker means "invalid", not this
# blanket degrade. INTENT_LIST_ALERTS and INTENT_END_VOICE_SESSION are
# also not here -- neither one is about any particular ticker at all.
_TICKER_SCOPED_INTENTS = {
    INTENT_TICKER_QUESTION, INTENT_PORTFOLIO_FIT, INTENT_FULL_REPORT_REQUEST, INTENT_CREATE_ALERT,
    INTENT_WATCHLIST_ADD, INTENT_WATCHLIST_REMOVE, INTENT_ADD_HOLDING,
}

_INTENT_RE = re.compile(r"INTENT:\s*(\w+)", re.IGNORECASE)

# A small, fast model is enough for an 8-way classification and a short
# generic reply -- same model_consensus.py precedent of picking a
# cheap model for a short, low-stakes call rather than the biggest
# available one. allam-2-7b specifically (not a gpt-oss/reasoning
# model): it answers directly with no hidden chain-of-thought tax, so
# it reliably fits this file's tight max_new_tokens budgets (20-200) --
# confirmed live against Groq's current catalog after llama-3.3-70b
# was fully removed (was 404ing every call here, degrading every
# message to INTENT_GENERAL by design -- see classify_intent's
# docstring -- which made this file look like it worked while actually
# never classifying anything). Public (not `_`-prefixed): see this
# constant's own cross-module reuse note further down.
CHAT_MODEL = "allam-2-7b"

# Classification specifically (not CHAT_MODEL above, which still backs
# every actual reply) needs a stronger model -- confirmed live over a
# real multi-turn voice session: allam-2-7b reliably ignores this
# file's "classify the LATEST message, not the history" instruction
# once real conversation history is present, and just repeats whatever
# intent the FIRST message in the session got for every message after
# it (e.g. "Justin, can you hear me?" right after a portfolio question
# also came back portfolio_status). A gpt-oss reasoning model normally
# can't be used here at all (see CHAT_MODEL's own comment -- its hidden
# chain-of-thought burns through a short token budget before it ever
# reaches the actual answer), but classification's own prompt is by far
# the shortest, most template-shaped output in this file -- a budget
# large enough for gpt-oss-20b to finish reasoning AND answer (400, not
# 20) measured under 1s per call live, and got every reproduced
# misclassification above right. Not applied to the other CHAT_MODEL
# call sites in this file (the actual reply-generation ones) -- those
# weren't confirmed broken, and each would need its own budget
# increase and re-verification to use a reasoning model safely; keeping
# this change scoped to the one call site actually proven broken.
_CLASSIFY_MODEL = "openai/gpt-oss-20b"
_CLASSIFY_MAX_TOKENS = 400

# How many recent messages (not turns -- individual user+assistant rows)
# to fold into a prompt as conversation memory. Small on purpose: this
# is a fast/cheap classification+synthesis call, not the full research
# pipeline, and a short window is enough to resolve "it"/"that" without
# growing the prompt (and the per-call cost/latency) unboundedly as a
# thread gets long.
MAX_HISTORY_MESSAGES = 6

# Shared persona/disclaimer fragments for every LLM-synthesis prompt
# below -- found live: repeating "never claim to be a licensed
# advisor" as an instruction made every single reply end with some
# variant of "consult a licensed financial advisor," even on messages
# where it added nothing. The compliance boundary itself doesn't
# change (still never a personalized buy/sell call) -- only the
# instruction to actively RESTATE it in every reply is gone, since the
# app already shows that disclaimer once, on the Chat page itself and
# in the global footer, not per-message. Public (not `_`-prefixed),
# same as CHAT_MODEL/suggest_quantity above -- also reused by
# app/reasoning/daily_briefing.py's own LLM-synthesis step, so the tone
# and compliance boundary stay identical across every conversational
# surface instead of a second, drifting copy of this exact wording.
PERSONA = (
    "You're FinSight, a knowledgeable financial analyst talking directly to the user -- clear, warm, and "
    "direct, the way you'd explain something to a colleague, not a compliance script."
)
DISCLAIMER_NOTE = (
    "Don't append a 'consult a licensed advisor' disclaimer -- that's already shown once elsewhere in the "
    "app; repeating it in every reply reads as evasive, not careful."
)

# ---------------------------------------------------------------- conversational order placement
#
# Foundation safety mechanic every order-placing feature in this module
# reuses: parse a command into a concrete, priced leg (or list of legs
# for a batch -- see Feature 1's expansion helpers), show the user
# EXACTLY what will happen, and only call db.execute_order on an
# explicit affirmative reply. Side/quantity/confirmation are all parsed
# deterministically (plain regex/word matching below), never guessed by
# the LLM -- same "never let the LLM spell out something that actually
# matters" reasoning company_resolver.py's own docstring gives for
# ticker resolution, extended here to the two other fields (how many
# shares, which direction) that would be genuinely dangerous to
# mis-hear, especially over voice (see Feature 7).
#
# Per-user pending-order state lives in this plain in-memory dict, not
# a new table -- this app already assumes a single SQLite-file/single-
# process deployment everywhere else (jobs.py's ThreadPoolExecutor
# being the clearest precedent), so there's no multi-instance scenario
# where one instance's pending order needs to be visible to another.
# A pending order is deliberately ephemeral: lost on a process
# restart just means the user re-issues the command, the same
# "no harm done" property any other in-flight, unconfirmed HTTP
# request already has. Each entry is
# {"legs": [{"ticker", "side", "quantity", "price", "currency", "rationale"}, ...]}
# -- always a list, even for a single order, so the batch feature
# (Feature 1) and the single-order path share one confirm/execute
# implementation instead of two parallel ones.
_pending_orders: dict = {}

_BUY_RE = re.compile(r"\bbuy\b", re.IGNORECASE)
_SELL_RE = re.compile(r"\bsell\b", re.IGNORECASE)
_QUANTITY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")
# Requires the "at"/"@" marker (unlike _QUANTITY_RE's bare first-number
# grab above) so _parse_avg_cost can pull a cost out of a message that
# ALSO states a quantity ("10 shares at $150") without colliding on the
# same number -- same [₹$]? + comma-tolerant shape as _ALERT_PRICE_RE
# below.
_AVG_COST_RE = re.compile(r"(?:\bat\b|@)\s*[₹$]?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)

# Feature 1 (batching) -- a SMALL FIXED set of exact phrases, not
# open-ended strategy language (see this module's own combined
# docstring section and the spec this was built from: "start narrow").
# Two independently-recognized surface phrasings resolve to the same
# "buy_top_n" batch type since the feature's own goal statement and its
# own build-list use two different example phrasings for the same idea.
_SELL_RATED_RE = re.compile(r"\bsell\s+everything\s+rated\s+(buy|hold|sell)\b", re.IGNORECASE)
_PUT_AMOUNT_TOP_N_RE = re.compile(
    r"put\s+[₹$]?\s*([\d,]+(?:\.\d+)?)\s*(?:usd|dollars?|inr|rupees|₹|\$)?\s*into\s+(?:my\s+)?top\s+(\d+)\s+buy-?rated\s+watchlist\s+stocks?",
    re.IGNORECASE,
)
_BUY_TOP_N_WITH_AMOUNT_RE = re.compile(
    r"buy\s+across\s+(?:my\s+)?top\s+(\d+)\s+buy-?rated\s+watchlist\s+stocks?\s+(?:with|using)\s+[₹$]?\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Feature 3 (price alerts) -- "X drops below/rises above PRICE" is the
# one narrow phrasing this recognizes (see classify_intent's own
# create_alert examples). The auto-execute opt-in is a SEPARATE keyword
# from side/direction on purpose -- see _parse_alert_command's own
# docstring for why that's a deliberately harder bar to clear than a
# plain alert.
_ALERT_PRICE_RE = re.compile(r"(above|below)\s+[₹$]?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_AUTO_OPT_IN_RE = re.compile(r"\bautomatically\b|\bauto[\s-]?execute\b", re.IGNORECASE)

# Deliberately exact-phrase matching (a whole normalized message, or a
# normalized message starting with one of these phrases), not a
# substring search -- "no way I'd sell more" must never be read as a
# "no" to an unrelated pending order just because "no" occurs in it
# somewhere. Whitespace-normalized and lowercased before comparison.
_AFFIRMATIVE_PHRASES = ("yes", "y", "yeah", "yep", "yup", "confirm", "confirmed", "do it", "place it", "go ahead", "sure", "ok", "okay")
_NEGATIVE_PHRASES = ("no", "n", "nope", "cancel", "nevermind", "never mind", "stop", "don't", "dont")


def _parse_side(message: str) -> "str | None":
    """BUY/SELL/None, deterministic word-boundary match -- whichever of
    "buy"/"sell" appears FIRST in the message if (unusually) both are
    present, since that's the one the user is actually acting on right
    now (e.g. "I already sold once today, buy 5 more AAPL")."""
    buy_match = _BUY_RE.search(message)
    sell_match = _SELL_RE.search(message)
    if buy_match and sell_match:
        return "BUY" if buy_match.start() < sell_match.start() else "SELL"
    if buy_match:
        return "BUY"
    if sell_match:
        return "SELL"
    return None


def _parse_quantity(message: str) -> "float | None":
    """The first standalone number in the message, or None if there
    isn't one (or it's zero/negative) -- Feature 4 is what fills this
    gap with a suggested quantity; this function's only job is
    detecting whether the user already gave one. Tickers here are
    always alphabetic (US/NSE symbols), so a bare number can't be
    mistaken for one."""
    match = _QUANTITY_RE.search(message)
    if not match:
        return None
    quantity = float(match.group(1))
    return quantity if quantity > 0 else None


def _parse_confirmation(message: str) -> "str | None":
    """"yes"/"no"/None for a reply to a pending-order confirmation
    question. None (neither a clear yes nor a clear no) is a real,
    expected outcome -- handle_chat_message() below treats that as
    "the user moved on", dropping the stale pending order rather than
    guessing which way an ambiguous reply meant."""
    # Commas folded into whitespace (not just stripped from the ends)
    # so "yes, place it" / "no, don't" still match "yes "/"no " below --
    # a trailing-only strip would miss punctuation right after the
    # leading word.
    normalized = re.sub(r"[,.!?]+", " ", message.strip().lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if any(normalized == p or normalized.startswith(p + " ") for p in _AFFIRMATIVE_PHRASES):
        return "yes"
    if any(normalized == p or normalized.startswith(p + " ") for p in _NEGATIVE_PHRASES):
        return "no"
    return None


def _pending_order_ticker(pending: dict) -> "str | None":
    """The ticker to carry over onto the assistant's chat_messages row
    (see add_chat_message's `ticker` column) -- only meaningful for a
    single-leg order; a batch has no one ticker to carry forward."""
    legs = pending["legs"]
    return legs[0]["ticker"] if len(legs) == 1 else None


def _execute_pending_order(user_id: str) -> str:
    """Calls db.execute_order for every leg of the user's pending order
    (always present when this is called -- handle_chat_message only
    calls it after confirming _pending_orders[user_id] exists), then
    clears the pending state regardless of outcome -- a failed leg must
    never leave a stale, already-shown proposal sitting around to be
    silently re-confirmed by a later, unrelated "yes". Reports exactly
    which legs filled and which didn't (Feature 1's own acceptance
    requirement) rather than aborting the whole batch on one failure or
    silently skipping the failure."""
    pending = _pending_orders.pop(user_id, None)
    if pending is None:
        return "There's no pending order to confirm."

    filled, failed = [], []
    for leg in pending["legs"]:
        try:
            db.execute_order(
                user_id, leg["ticker"], leg["side"], leg["quantity"], leg["price"], leg["currency"],
                rationale=leg.get("rationale"),
            )
            filled.append(leg)
        except ValueError as e:
            failed.append((leg, str(e)))

    if len(pending["legs"]) == 1:
        if filled:
            leg = filled[0]
            symbol = currency_symbol(leg["currency"])
            return f"Done -- {leg['side']} {leg['quantity']:g} {leg['ticker']} filled at {symbol}{leg['price']:,.2f}."
        return f"Couldn't place that order: {failed[0][1]}"

    lines = []
    if filled:
        lines.append("Filled: " + ", ".join(f"{l['side']} {l['quantity']:g} {l['ticker']}" for l in filled) + ".")
    if failed:
        lines.append("Failed: " + ", ".join(f"{l['ticker']} ({err})" for l, err in failed) + ".")
    return " ".join(lines)


def _parse_batch_command(message: str) -> "dict | None":
    """Recognizes the small fixed set of batch phrasings this feature
    supports -- None for anything else, including a single-ticker order
    (that's the non-batch path _handle_place_order already has) or any
    open-ended portfolio-strategy sentence this deliberately does NOT
    try to parse."""
    m = _SELL_RATED_RE.search(message)
    if m:
        return {"type": "sell_rated", "rating": m.group(1).capitalize()}

    m = _PUT_AMOUNT_TOP_N_RE.search(message)
    if m:
        return {"type": "buy_top_n", "amount": float(m.group(1).replace(",", "")), "n": int(m.group(2))}

    m = _BUY_TOP_N_WITH_AMOUNT_RE.search(message)
    if m:
        return {"type": "buy_top_n", "n": int(m.group(1)), "amount": float(m.group(2).replace(",", ""))}

    return None


def _expand_sell_rated_batch(user_id: str, rating: str) -> "tuple[list, list]":
    """(legs, notes) for "sell everything rated {rating}" -- one SELL
    leg per currently-held ticker whose latest completed rating matches
    exactly, sized at the full held quantity. A holding that matches
    the rating but whose live quote lookup fails is left OUT of legs
    (never priced with a guess) and explained in notes instead --
    same degrade-honestly discipline as the single-order path."""
    legs, notes = [], []
    for h in db.get_portfolio_holdings(user_id):
        ticker = h["ticker"]
        if db.get_latest_rating_for_ticker(user_id, ticker) != rating:
            continue
        try:
            quote = get_quote(ticker)
        except Exception:
            notes.append(f"couldn't check {ticker}'s price, skipped")
            continue
        legs.append({
            "ticker": ticker, "side": "SELL", "quantity": h["quantity"],
            "price": quote["price"], "currency": quote.get("currency", "USD"),
            "rationale": f"Batch: sell everything rated {rating}",
        })
    return legs, notes


def _expand_buy_top_n_batch(user_id: str, n: int, amount: float) -> "tuple[list, list]":
    """(legs, notes) for "buy across top N Buy-rated watchlist stocks"
    (any of this feature's two recognized phrasings). "Top" means the
    first N Buy-rated tickers in the watchlist's own existing order
    (most-recently-added first, see db.get_watchlist) -- not a
    re-ranking by upside/fair-value, which would need a fresh per-
    ticker lookup this feature deliberately stays narrow enough to
    avoid. `amount` is split evenly across however many qualifying
    tickers were ACTUALLY found (which may be fewer than N -- noted,
    not silently under-spent) and spent in each ticker's own native
    currency, never converted -- same "never mix currencies" discipline
    build_portfolio_view's own docstring already commits to."""
    candidates = [
        w["ticker"] for w in db.get_watchlist(user_id)
        if db.get_latest_rating_for_ticker(user_id, w["ticker"]) == "Buy"
    ][:n]
    if not candidates:
        return [], []

    notes = []
    if len(candidates) < n:
        notes.append(f"only {len(candidates)} Buy-rated watchlist stock{'s' if len(candidates) != 1 else ''} found (asked for {n})")

    per_leg_amount = amount / len(candidates)
    legs = []
    for ticker in candidates:
        try:
            quote = get_quote(ticker)
        except Exception:
            notes.append(f"couldn't check {ticker}'s price, skipped")
            continue
        price = quote["price"]
        legs.append({
            "ticker": ticker, "side": "BUY", "quantity": round(per_leg_amount / price, 4),
            "price": price, "currency": quote.get("currency", "USD"),
            "rationale": f"Batch: buy across top {n} Buy-rated watchlist stocks",
        })
    return legs, notes


# Feature 4 (position sizing) -- a fixed, statable-in-one-sentence rule:
# a percentage of current total portfolio value, scaled by the user's
# own saved risk_tolerance (PATCH /v1/auth/risk-tolerance). Deliberately
# simple/transparent, not a black box -- the confirmation turn always
# names the exact percentage and risk level that produced the number,
# same "one sentence" requirement the spec this was built from states
# explicitly.
_RISK_TOLERANCE_SIZING_PCT = {"Conservative": 0.02, "Moderate": 0.05, "Aggressive": 0.10}
_DEFAULT_RISK_TOLERANCE = "Moderate"  # same fallback main.py's own GET /v1/auth/me already uses


def suggest_quantity(user_id: str, ticker: str, price: float) -> "tuple[float, str] | None":
    """(quantity, one-sentence sizing explanation) for a BUY with no
    stated quantity, or None when there's genuinely nothing real to
    size off of (an empty/unpriced portfolio) -- never a fabricated
    fallback number, same degrade-honestly discipline as everywhere
    else in this module. Public (not `_`-prefixed): also reused by
    app/reasoning/price_alerts.py to size a BUY-side auto-execute price
    alert at trigger time, the same sizing rule either way."""
    user = db.get_user_by_id(user_id)
    risk_tolerance = (user.get("risk_tolerance") if user else None) or _DEFAULT_RISK_TOLERANCE
    pct = _RISK_TOLERANCE_SIZING_PCT.get(risk_tolerance, _RISK_TOLERANCE_SIZING_PCT[_DEFAULT_RISK_TOLERANCE])

    total_value = build_portfolio_view(user_id)["summary"].get("total_market_value")
    if not total_value:
        return None

    quantity = round((total_value * pct) / price, 4)
    if quantity <= 0:
        return None

    return quantity, f"that's about {pct * 100:g}% of your portfolio, sized for {risk_tolerance}"


def _propose_batch_order(user_id: str, batch: dict) -> str:
    """Turn 1 of the batch confirm-then-execute flow -- expands the
    recognized batch type into concrete priced legs and stores them as
    this user's pending order, same _pending_orders shape (and same
    _execute_pending_order path) a single order uses. The confirmation
    always shows the FULL expanded list (this feature's own explicit
    acceptance requirement) -- never a blanket "confirm the batch?" for
    a list the user hasn't actually seen yet."""
    if batch["type"] == "sell_rated":
        legs, notes = _expand_sell_rated_batch(user_id, batch["rating"])
        side_word, empty_message = "sell", f"You don't currently hold anything rated {batch['rating']}."
    else:
        legs, notes = _expand_buy_top_n_batch(user_id, batch["n"], batch["amount"])
        side_word, empty_message = "buy", "Couldn't find any Buy-rated watchlist stocks to buy."

    if not legs:
        note_bit = (" (" + "; ".join(notes) + ")") if notes else ""
        return empty_message + note_bit

    _pending_orders[user_id] = {"legs": legs}

    leg_list = ", ".join(f"{l['quantity']:g} {l['ticker']}" for l in legs)
    note_bit = (" Note: " + "; ".join(notes) + ".") if notes else ""
    return (
        f"This will {side_word}: {leg_list}.{note_bit} "
        "Confirm all? Reply \"yes\" to place all orders or \"no\" to cancel."
    )


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
    # Confirmed live: allam-2-7b (this file's classifier model -- see
    # CHAT_MODEL's own docstring for why a bigger/reasoning model isn't
    # an option here) tends to just repeat whatever intent the FIRST
    # message in a session got, for every message after it, once real
    # conversation history is present -- e.g. "Justin, can you hear me?"
    # right after a portfolio question got classified portfolio_status
    # too, even though it has nothing to do with a portfolio. The
    # original prompt already said "classify the LATEST message, not
    # this history" once, at the top of the history block; that alone
    # wasn't enough for this model to actually follow it. The explicit
    # "do NOT let the topic ... influence your choice" framing below,
    # repeated again right next to LATEST MESSAGE, measurably fixed this
    # for real reproduced cases -- weak models need the instruction
    # repeated near where it actually applies, not just stated once
    # up front.
    return f"""Classify ONLY the LATEST MESSAGE below into EXACTLY ONE of these categories. Do NOT let the topic of RECENT CONVERSATION influence your choice -- it is provided only so you can resolve pronouns like "that"/"it" in the latest message, nothing else. A new message on a completely different topic must be classified on its own content, even if the conversation was previously about something else.

portfolio_status - asking about their overall portfolio, holdings, or how it's doing
ticker_question - asking what's going on with, or for information about, a specific stock/company (not asking for a deep valuation report)
portfolio_fit - asking whether/how a specific stock fits, overlaps with, or diversifies their existing portfolio
full_report_request - explicitly asking for a full/deep/detailed research report, DCF, or valuation
advice_request - asking where/how to invest, allocate, or spend money based on their own goals or situation (e.g. "where should I invest", "how should I spend $40k", "what should a beginner do") -- NOT asking about an existing holding's status
stock_discovery_request - asking the assistant to find, discover, suggest, or recommend NEW specific stocks/companies it hasn't already named (e.g. "find me two new companies", "give me some stock ideas", "what should I buy next") -- NOT asking about a ticker already mentioned in this conversation, and NOT general allocation advice
place_order - an actual instruction to buy or sell a stock right now (e.g. "buy 5 AAPL", "sell 10 TCS", "sell everything rated Sell") -- a trade command, NOT a question about whether/what to trade (that's advice_request or ticker_question)
create_alert - setting up a standing price alert to check later (e.g. "alert me if AAPL drops below 180", "sell if TCS drops below 4000", "notify me when MSFT rises above 300") -- a WATCH for later, NOT an immediate buy/sell (that's place_order)
list_alerts - asking what price alerts they currently have set (e.g. "what alerts do I have", "show my price alerts")
watchlist_add - adding a specific stock/company to their watchlist to track it going forward, with NO price condition and NO buy/sell instruction (e.g. "add Bajaj Finance to my watchlist", "put TCS on my watchlist", "start tracking AAPL", "watchlist NVDA") -- a passive tracking request, NOT a price alert (that's create_alert, which always names a trigger price like "below 180"), NOT a trade (that's place_order), and NOT asking for the stock's current status (that's ticker_question)
watchlist_remove - removing a specific stock/company from their watchlist (e.g. "remove TCS from my watchlist", "stop tracking AAPL", "take Bajaj Finance off my watchlist")
add_holding - recording a stock they ALREADY OWN into their portfolio at a cost basis they state themselves (self-reported, not a live-price trade) (e.g. "add 10 shares of AAPL to my portfolio at $150", "I already own 5 TCS at 4000, add it to my portfolio") -- NOT a new trade at today's price (that's place_order) and NOT tracking-only with no stated ownership/cost (that's watchlist_add)
daily_briefing - asking for a pulled-together summary/digest of portfolio performance, recent rating changes, AND upcoming events all at once (e.g. "give me my briefing", "what's my daily update", "catch me up") -- NOT a plain "how's my portfolio doing" (that's portfolio_status, which has no rating-change/upcoming-event digest)
end_voice_session - explicitly asking to end/stop the current voice session or stop listening (e.g. "end the session", "stop listening", "that's all for now", "goodbye") -- NOT logging out of the account or deleting anything, just stopping this hands-free conversation
general - anything else, including generic finance chit-chat, greetings, small talk, or a message unrelated to the categories above

{ticker_note}

RECENT CONVERSATION (context only, for pronoun resolution -- do NOT classify based on its topic):
{_format_history(history)}

LATEST MESSAGE (classify only this): {message}

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
        provider = HostedProvider(model=_CLASSIFY_MODEL)
        raw = provider.generate(_build_classify_prompt(message, ticker is not None, history), max_new_tokens=_CLASSIFY_MAX_TOKENS)
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
        f"{PERSONA} Use ONLY the real data below -- never invent numbers. Answer the user's actual question in "
        "2-4 sentences: if they ask about diversification, reference the sector mix; if they ask what's going "
        "well/badly, reference the per-holding ratings and overall P&L; if the question is generic, summarize "
        "total value and P&L. If the message references something from earlier in the conversation (e.g. "
        "'that', 'it'), use the recent conversation below to understand what they mean. Never name a buy/sell "
        f"call beyond restating the existing rating. {DISCLAIMER_NOTE}\n\n"
        f"PORTFOLIO DATA:\n{portfolio_data}\n\n"
        f"RECENT CONVERSATION:\n{_format_history(history)}\n\nMESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=200).strip()
    except LLMProviderError:
        return portfolio_data


def _handle_ticker_question(ticker: str, message: str, history: list) -> str:
    """Same "raw data doubles as LLM grounding AND outage fallback"
    pattern as _handle_portfolio_status -- a plain "what's going on
    with X" and a follow-up like "what's your source for that" or
    "why do you say that" both need real answers, not the identical
    canned rating dump repeated verbatim regardless of what was asked
    (a real bug caught live: a user's "what's your source, where did
    you get this" follow-up got back the exact same rating summary as
    the question before it)."""
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

    stock_data = "\n".join(lines)

    prompt = (
        f"{PERSONA} Use ONLY the real data below -- never invent numbers. Answer the user's actual question in "
        "2-4 sentences: if they ask for the source/methodology, explain this is FinSight's own algorithmic "
        "rating (computed from a DCF valuation plus financial-health and momentum/growth signals), not a "
        "third-party analyst call; if they ask why a flag applies, reference the flags below; if the question "
        "is generic, summarize the rating and news. If the message references something from earlier in the "
        "conversation, use the recent conversation below to understand what they mean. Never name a buy/sell "
        f"call beyond restating the existing rating. {DISCLAIMER_NOTE}\n\n"
        f"STOCK DATA:\n{stock_data}\n\n"
        f"RECENT CONVERSATION:\n{_format_history(history)}\n\nMESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=200).strip()
    except LLMProviderError:
        return stock_data


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
        f"{PERSONA} You're answering a request for investing/allocation guidance. Use ONLY the real context "
        "below -- never invent numbers. Answer in 2-4 sentences, at the level of asset classes/sectors/"
        "allocation percentages, not a specific buy/sell call on an individual stock. If onboarding preferences "
        "are unset, give general starter guidance (e.g. diversified index funds first) but mention completing "
        "onboarding for guidance tailored to their risk tolerance. If the message references something from "
        "earlier in the conversation (e.g. 'that', 'the $40k thing'), use the recent conversation below to "
        f"understand what they mean. {DISCLAIMER_NOTE}\n\n"
        "CONTEXT:\n" + "\n".join(context_lines)
        + f"\n\nRECENT CONVERSATION:\n{_format_history(history)}\n\nMESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=200).strip()
    except LLMProviderError:
        return "I can give allocation-level guidance using your real portfolio and preferences, but couldn't generate a response just now -- try again in a moment."


def _handle_stock_discovery_request() -> str:
    """Deliberately a FIXED string, not an LLM call -- caught live: a
    user pushing on "find me two new companies" got a different,
    self-contradicting answer every turn (first "I'll suggest two
    sectors" using their real portfolio data, then "I would need to
    know more about your current portfolio" moments later, having just
    used it). Recommending brand-new, unresearched securities is a real
    capability gap, not something to paper over with an LLM guessing at
    a plausible-sounding deflection each time -- one honest, consistent
    answer beats a different vague one on every turn.

    Points to the Screener (real filters over the Tracked Universe --
    market cap, valuation, dividend yield, volume) as the actual way to
    discover something new, rather than a flat refusal with nowhere to
    go -- deterministic filtering over real data is a fundamentally
    different (and fine) thing from an LLM naming unresearched picks,
    which is the specific capability this function declines above."""
    return (
        "I can't discover or recommend brand-new stocks -- picking specific, unresearched securities is "
        "outside what this app does (same boundary as real estate: allocation-level guidance only, never "
        "property- or security-specific). What I can do: take you to the Screener to filter real stocks by "
        "market cap, valuation, dividend yield, or volume, tell you how a stock you already have in mind fits "
        "your portfolio, or pull a full research report for a specific ticker you name."
    )


def _handle_place_order(user_id: str, ticker: str, message: str, history: list) -> str:
    """Turn 1 of the confirm-then-execute flow: parse a single-ticker
    order out of `message`, validate it against real data (a live
    quote, and -- for a SELL -- the user's actual holding), and stash
    it as this user's pending order. Returns the confirmation question
    itself; NEVER calls db.execute_order (see handle_chat_message,
    which is the only place a "yes" reply does that).

    Ticker resolution is already done by classify_intent() (the same
    resolve_companies()-first, history-carry-over-second mechanism
    every other ticker-scoped intent in this module uses) -- this
    function's own job is just the two fields classify_intent doesn't
    resolve: side and quantity, both parsed deterministically above,
    never guessed by the LLM.

    A message with no ticker at all falls through to here (place_order
    is deliberately not in _TICKER_SCOPED_INTENTS) so a batch command
    like "sell everything rated Sell" can still be handled -- checked
    FIRST, below, before `ticker` is ever consulted (a batch command
    legitimately names no single ticker, and any carried-over ticker
    from earlier conversation would be irrelevant to it anyway).

    A message that EXPLICITLY names something that fails to resolve
    must never fall back to guessing a different ticker discussed
    earlier -- confirmed live: "buy 10 ZZZZ" (an unresolvable ticker)
    silently proposed buying a completely different, unrelated ticker
    from an earlier turn instead of failing clearly. Checked via
    has_unresolved_ticker_attempt(), NOT a bare
    `not resolve_companies(message)` truthiness check -- this handler
    has its own real two-turn sub-flow (the no-quantity "how many
    shares of {ticker} would you like to buy?" prompt below) that
    deliberately expects a ticker-LESS reply like "buy 10" or "10",
    relying on the exact same history-carryover classify_intent()
    already did. A blanket resolve_companies(message) truthiness check
    can't tell "buy 10 ZZZZ" (a real, failed attempt) apart from
    "buy 10" (names no ticker at all, carryover is correct) -- both
    return [] alike -- so it would silently break that legitimate
    continuation. has_unresolved_ticker_attempt() only fires on an
    actual leftover ticker-shaped token, not on a message that simply
    omits one."""
    batch = _parse_batch_command(message)
    if batch is not None:
        return _propose_batch_order(user_id, batch)

    if not ticker or has_unresolved_ticker_attempt(message):
        return "I couldn't tell which stock you mean -- try naming a ticker or company, e.g. \"buy 5 AAPL\"."

    side = _parse_side(message)
    if side is None:
        return "I couldn't tell if that's a buy or a sell -- try \"buy 5 AAPL\" or \"sell 5 AAPL\"."

    quantity = _parse_quantity(message)

    # A SELL with no stated quantity still just asks -- Feature 4's
    # suggestion only ever applies to a BUY (a "how much should I sell"
    # default would be a much riskier guess than "how much should I
    # buy", and "sell everything ..." is already the batch feature's
    # own explicit phrasing for that case). Checked before the quote
    # fetch below so a doomed SELL never spends a network call.
    if quantity is None and side == "SELL":
        return f"How many shares of {ticker} would you like to sell? (e.g. \"sell 5 {ticker}\")"

    if side == "SELL":
        holdings = {h["ticker"]: h["quantity"] for h in db.get_portfolio_holdings(user_id)}
        held = holdings.get(ticker, 0.0)
        if quantity > held:
            if held == 0:
                return f"You don't currently hold any {ticker}, so there's nothing to sell."
            return f"You only hold {held:g} share{'s' if held != 1 else ''} of {ticker} -- can't sell {quantity:g}."

    try:
        quote = get_quote(ticker)
    except Exception:
        # Guardrail: degrade honestly rather than fabricate a price or
        # silently drop the command -- same "None on failure, never a
        # guess" discipline as the rest of this app (WACC floor/NaN
        # handling/options pricer).
        return f"I couldn't check {ticker}'s price right now -- try again in a moment."

    price = quote["price"]
    currency = quote.get("currency", "USD")
    symbol = currency_symbol(currency)

    sizing_note = None
    if quantity is None:  # side == "BUY" -- SELL already returned above
        suggestion = suggest_quantity(user_id, ticker, price)
        if suggestion is None:
            return f"How many shares of {ticker} would you like to buy? (e.g. \"buy 5 {ticker}\")"
        quantity, sizing_note = suggestion

    rationale = "User-initiated via chat" if sizing_note is None else f"User-initiated via chat, sized via suggestion ({sizing_note})"
    _pending_orders[user_id] = {
        "legs": [{"ticker": ticker, "side": side, "quantity": quantity, "price": price, "currency": currency, "rationale": rationale}],
    }

    if sizing_note is not None:
        # Confirming or overriding the suggestion is still an explicit
        # step every time -- this IS the same confirmation turn as the
        # explicit-quantity path below, just with the suggested number
        # and its one-sentence justification spelled out. A "yes" reply
        # confirms it exactly like any other pending order; a reply
        # with a different quantity (e.g. "buy 10 AAPL instead") isn't
        # a yes/no, so handle_chat_message's own ambiguous-reply path
        # drops this suggestion and re-parses that as a fresh command.
        return (
            f"Suggest {quantity:g} shares of {ticker} ({sizing_note}) at today's price of {symbol}{price:,.2f} -- "
            "confirm, or tell me a different quantity."
        )

    return (
        f"Confirm: {side} {quantity:g} {ticker} at today's price of {symbol}{price:,.2f}? "
        "Reply \"yes\" to place the order or \"no\" to cancel."
    )


def _parse_alert_command(message: str) -> "dict | None":
    """{"direction", "target_price", "side", "auto_execute"} for the
    one narrow phrasing this recognizes ("X drops below/rises above
    PRICE"), or None. `side` defaults to SELL when the message names
    neither buy nor sell (e.g. "alert me if AAPL drops below 180" --
    a passive, protective heads-up reads most naturally as a SELL-side
    watch). `auto_execute` is True ONLY when BOTH an explicit side AND
    a separately-worded auto opt-in ("automatically"/"auto-execute")
    are present -- a deliberately harder bar to clear than a plain
    alert, never the default reading of one ambiguous command (this
    feature's own guardrail)."""
    m = _ALERT_PRICE_RE.search(message)
    if not m:
        return None
    direction = m.group(1).lower()
    target_price = float(m.group(2).replace(",", ""))
    side = _parse_side(message)
    auto_execute = side is not None and bool(_AUTO_OPT_IN_RE.search(message))
    return {"direction": direction, "target_price": target_price, "side": side or "SELL", "auto_execute": auto_execute}


def _handle_create_alert(user_id: str, ticker: str, message: str, history: list) -> str:
    """Registers a standing price alert and confirms it -- unlike
    place_order, this is a low-stakes standing watch (not itself a
    trade), so it takes effect immediately with no separate confirm
    turn; the one genuinely risky case (auto_execute=1, a FUTURE
    unattended trade) is instead gated by requiring the separately-
    worded opt-in _parse_alert_command demands, not a second turn
    here.

    `ticker` must come from resolve_companies() on THIS message, not
    classify_intent's history-carryover fallback -- same bug class
    confirmed live for place_order/watchlist_add ("buy 10 ZZZZ" /
    "add asdkfjaslkdfj" silently acting on an unrelated earlier
    ticker instead of failing clearly). Arguably higher-stakes here
    than place_order: an auto_execute alert is a FUTURE unattended
    trade the user won't be watching for, so setting it on the wrong
    ticker could go unnoticed far longer than an immediate mistaken
    buy."""
    if not ticker or not resolve_companies(message):
        return "I couldn't tell which stock you mean -- try naming a ticker or company, e.g. \"alert me if AAPL drops below 180\"."

    parsed = _parse_alert_command(message)
    if parsed is None:
        return f"I couldn't tell what price condition you mean -- try \"alert me if {ticker} drops below 180\"."

    db.create_price_alert(
        user_id, ticker, parsed["side"], parsed["direction"], parsed["target_price"], parsed["auto_execute"],
    )

    try:
        currency = get_quote(ticker).get("currency", "USD")
    except Exception:
        currency = "USD"  # degrade -- the SET price still displays correctly, just without a live-confirmed symbol
    symbol = currency_symbol(currency)

    verb = "drops below" if parsed["direction"] == "below" else "rises above"
    action_bit = f"automatically {parsed['side'].lower()}" if parsed["auto_execute"] else "let you know"
    return f"Alert set: if {ticker} {verb} {symbol}{parsed['target_price']:,.2f}, I'll {action_bit} (simulated)."


def _handle_watchlist_add(user_id: str, ticker: str, message: str) -> str:
    """Adds a ticker to the watchlist and confirms it -- same single-
    turn, no-confirmation shape as _handle_create_alert above (a
    reversible, non-monetary action, not a trade). The get_quote()
    call below is enrichment only, never a gate: the ticker reaching
    this handler already passed resolve_companies() inside
    classify_intent(), and db.add_watchlist_item is INSERT OR IGNORE
    with no dependency on live market data, so a quote hiccup should
    never block a legitimate add -- same reasoning _handle_create_alert
    already applies to its own currency-symbol lookup.

    Confirmed live: classify_intent()'s ticker can come from
    _most_recent_ticker(history) when resolve_companies(message) finds
    nothing in the CURRENT message -- exactly right for a follow-up
    question ("is it a good buy?"), but wrong here: "remove ZZZZ from
    my watchlist" (an unresolvable ticker) silently removed a
    completely different, unrelated ticker from two turns earlier
    instead of failing clearly. Re-running resolve_companies on just
    this message (not trusting classify_intent's ticker blindly) is
    the deterministic way to tell "this message named a real company"
    apart from "classify_intent guessed from stale context" -- a write
    action naming something that fails to resolve must surface as
    "couldn't identify it", never silently act on whatever was talked
    about earlier."""
    if not ticker or not resolve_companies(message):
        return "I couldn't tell which stock you mean -- try naming a ticker or company, e.g. \"add AAPL to my watchlist\"."

    already_present = ticker in {item["ticker"] for item in db.get_watchlist(user_id)}
    db.add_watchlist_item(user_id, ticker)

    if already_present:
        return f"{ticker} is already on your watchlist."

    try:
        quote = get_quote(ticker)
        symbol = currency_symbol(quote.get("currency", "USD"))
        return f"Added {ticker} to your watchlist -- trading at {symbol}{quote['price']:,.2f} right now."
    except Exception:
        return f"Added {ticker} to your watchlist."  # degrade -- the add itself already succeeded above


def _handle_watchlist_remove(user_id: str, ticker: str, message: str) -> str:
    """Removes a ticker from the watchlist and confirms it. No quote
    lookup needed -- removal carries no price to report. Same
    resolve_companies(message)-must-match-the-CURRENT-message
    requirement as _handle_watchlist_add above, for the identical
    reason -- confirmed live against the identical failure mode."""
    if not ticker or not resolve_companies(message):
        return "I couldn't tell which stock you mean -- try naming a ticker or company, e.g. \"remove AAPL from my watchlist\"."

    was_present = ticker in {item["ticker"] for item in db.get_watchlist(user_id)}
    db.remove_watchlist_item(user_id, ticker)

    if not was_present:
        return f"{ticker} wasn't on your watchlist."
    return f"Removed {ticker} from your watchlist."


def _parse_avg_cost(message: str) -> "float | None":
    """The per-share cost from a phrase like "at $150" or "@ 1090" --
    deliberately requires the "at"/"@" marker (unlike _parse_quantity's
    bare first-number grab) so a quantity and a cost can coexist in the
    SAME message ("add 10 shares of AAPL at $150") without both regexes
    matching the same number."""
    match = _AVG_COST_RE.search(message)
    if not match:
        return None
    cost = float(match.group(1).replace(",", ""))
    return cost if cost > 0 else None


def _handle_add_holding(user_id: str, ticker: str, message: str, history: list) -> str:
    """Manual portfolio entry ("add 10 shares of AAPL to my portfolio
    at $150") -- a SELF-REPORTED position at a cost basis the user
    states, not a simulated trade at a live quote (that's
    place_order's job). Same shape as POST /v1/portfolio's own "+ Add
    Holding" flow, just reached via chat.

    `ticker` must come from resolve_companies() on THIS message, not
    classify_intent's history-carryover fallback -- same fix as
    place_order/create_alert/watchlist_add (see _handle_place_order's
    own docstring for the confirmed live bug this guards against).

    Refuses to silently overwrite an existing holding -- db.upsert_
    portfolio_holding REPLACES quantity/avg_cost rather than
    accumulating (see its own docstring: this is a current-position
    editor, not a transaction ledger), so blindly calling it again for
    a ticker already held would silently discard the user's real,
    previously-recorded cost basis. Directing them to the Portfolio
    card instead of guessing how to combine the two is the same
    "degrade honestly rather than fabricate" discipline the rest of
    this app already uses for a failed quote/price lookup."""
    if not ticker or has_unresolved_ticker_attempt(message):
        return "I couldn't tell which stock you mean -- try naming a ticker or company, e.g. \"add 10 shares of AAPL to my portfolio at $150\"."

    existing = {h["ticker"]: h for h in db.get_portfolio_holdings(user_id)}
    if ticker in existing:
        h = existing[ticker]
        try:
            currency = get_quote(ticker).get("currency", "USD")
        except Exception:
            currency = "USD"
        symbol = currency_symbol(currency)
        return (
            f"You already hold {h['quantity']:g} shares of {ticker} at an average cost of "
            f"{symbol}{h['avg_cost']:,.2f} -- to change that, edit it from the Portfolio card instead of "
            "adding it again, so the existing record isn't lost."
        )

    # The avg-cost number is cut out of the text handed to _parse_quantity
    # first -- confirmed by a test catching it live: "add AAPL to my
    # portfolio at $150" has only ONE number in the whole message, so
    # _parse_quantity's own bare first-number grab would otherwise read
    # the cost's own "150" as the QUANTITY too. Only matters when a
    # quantity is genuinely absent; a message stating both ("10 shares
    # ... at $150") already has two distinct numbers, so this is a no-op
    # for the common case.
    avg_cost_match = _AVG_COST_RE.search(message)
    quantity_text = message[:avg_cost_match.start()] + message[avg_cost_match.end():] if avg_cost_match else message
    quantity = _parse_quantity(quantity_text)
    avg_cost = _parse_avg_cost(message)
    if quantity is None or avg_cost is None:
        missing = (
            "how many shares and your average cost per share" if quantity is None and avg_cost is None
            else "how many shares" if quantity is None else "your average cost per share"
        )
        return f"Tell me {missing} for {ticker}, e.g. \"10 shares at $150\"."

    try:
        currency = get_quote(ticker).get("currency", "USD")
    except Exception:
        currency = "USD"  # degrade -- still records the holding, just without a live-confirmed currency symbol

    db.upsert_portfolio_holding(user_id, ticker, quantity, avg_cost)
    symbol = currency_symbol(currency)
    return f"Added {quantity:g} shares of {ticker} to your portfolio at an average cost of {symbol}{avg_cost:,.2f}."


def _handle_list_alerts(user_id: str) -> str:
    alerts = db.list_price_alerts(user_id)
    if not alerts:
        return "You don't have any active price alerts."

    lines = ["Your active price alerts:"]
    for a in alerts:
        verb = "drops below" if a["direction"] == "below" else "rises above"
        auto_bit = " (auto)" if a["auto_execute"] else ""
        lines.append(f"- {a['ticker']} {verb} {a['target_price']:,.2f} -> {a['side']}{auto_bit}")
    return "\n".join(lines)


def _handle_daily_briefing(user_id: str) -> dict:
    """{"text", "ticker"} for the "give me my briefing" intent --
    handled OUTSIDE the normal _HANDLERS dispatch (see
    handle_chat_message's own special-case branch) because, unlike
    every other intent, its `ticker` has to come from the briefing's
    OWN content (the one rating-changed ticker it names, if any), not
    from classify_intent's pre-pass over the bare word "briefing".

    Delegates the actual composition to
    app/reasoning/daily_briefing.py's build_briefing -- imported here,
    not at module level, since that module imports CHAT_MODEL/PERSONA/
    etc. FROM this one; a module-level import back the other way would
    be a circular import (same reasoning jobs.py's own factory
    functions defer their heavy imports for, just here to break a
    cycle rather than to skip an expensive one)."""
    from app.reasoning.daily_briefing import build_briefing

    briefing = build_briefing(user_id)
    # Advances the same "since" cursor a sweep-sent briefing would --
    # see db.set_last_briefing_at's own docstring for why an on-demand
    # ask counts exactly the same as a sweep delivery.
    db.set_last_briefing_at(user_id, time.time())
    return briefing


def _handle_end_voice_session() -> str:
    """A fixed string, not an LLM call -- there's nothing to generate,
    and the frontend keys off `intent == end_voice_session` specifically
    (not this text) to actually stop listening, so the wording here is
    purely what gets spoken/shown, never parsed."""
    return "Ending the session -- just say \"Hey FinSight\" or tap Start whenever you want to talk again."


def _handle_general(message: str, history: list) -> str:
    """Short, caveated LLM answer with no specific data context -- no
    portfolio/ticker to key off for this message, so this deliberately
    stays generic rather than fabricating specifics. Still gets the
    recent conversation, purely so a reference to an earlier turn reads
    as understood rather than ignored -- it has no new data to answer
    with regardless."""
    prompt = (
        f"{PERSONA} Answer the user's message in 1-3 sentences. You have no access to their portfolio or any "
        "specific stock data for this message -- if the question needs that, say so and suggest asking about a "
        "specific ticker or their portfolio instead. Use the recent conversation below only to understand "
        f"references like 'that' or 'it'. Never give personalized investment advice. {DISCLAIMER_NOTE}\n\n"
        f"RECENT CONVERSATION:\n{_format_history(history)}\n\nMESSAGE: {message}"
    )
    try:
        provider = HostedProvider(model=CHAT_MODEL)
        return provider.generate(prompt, max_new_tokens=150).strip()
    except LLMProviderError:
        return "I can help with your portfolio or a specific ticker -- try asking about one of those."


_HANDLERS = {
    INTENT_PORTFOLIO_STATUS: lambda user_id, ticker, message, history: _handle_portfolio_status(user_id, message, history),
    INTENT_TICKER_QUESTION: lambda user_id, ticker, message, history: _handle_ticker_question(ticker, message, history),
    INTENT_PORTFOLIO_FIT: lambda user_id, ticker, message, history: _handle_portfolio_fit(user_id, ticker),
    INTENT_FULL_REPORT_REQUEST: lambda user_id, ticker, message, history: _handle_full_report_request(ticker),
    INTENT_ADVICE_REQUEST: lambda user_id, ticker, message, history: _handle_advice_request(user_id, message, history),
    INTENT_STOCK_DISCOVERY_REQUEST: lambda user_id, ticker, message, history: _handle_stock_discovery_request(),
    INTENT_PLACE_ORDER: lambda user_id, ticker, message, history: _handle_place_order(user_id, ticker, message, history),
    INTENT_CREATE_ALERT: lambda user_id, ticker, message, history: _handle_create_alert(user_id, ticker, message, history),
    INTENT_LIST_ALERTS: lambda user_id, ticker, message, history: _handle_list_alerts(user_id),
    INTENT_WATCHLIST_ADD: lambda user_id, ticker, message, history: _handle_watchlist_add(user_id, ticker, message),
    INTENT_WATCHLIST_REMOVE: lambda user_id, ticker, message, history: _handle_watchlist_remove(user_id, ticker, message),
    INTENT_ADD_HOLDING: lambda user_id, ticker, message, history: _handle_add_holding(user_id, ticker, message, history),
    INTENT_END_VOICE_SESSION: lambda user_id, ticker, message, history: _handle_end_voice_session(),
    INTENT_GENERAL: lambda user_id, ticker, message, history: _handle_general(message, history),
}


def handle_chat_message(user_id: str, message: str, history: list = None) -> dict:
    """{"reply": str, "intent": str, "ticker": str | None} -- never
    raises; every handler already degrades gracefully on its own, and
    classify_intent() itself falls back to INTENT_GENERAL on failure.

    `history` is GET /v1/chat/history's own list shape (oldest first,
    from db.list_chat_messages) -- the caller's job to fetch it BEFORE
    persisting the current message, so it never includes the message
    being classified right now.

    Checks for a pending order (this user's own prior _handle_place_order
    proposal) BEFORE running intent classification at all -- turn 2 of
    the confirm-then-execute flow is a deterministic yes/no parse, never
    an LLM classification, since misclassifying a confirmation reply
    would be exactly the kind of mistake this whole flow exists to
    prevent. A reply that's neither a clear yes nor a clear no drops the
    stale pending order and falls through to normal classification for
    THIS message -- the user moved on, and leaving the old proposal
    alive risks it being accidentally confirmed later by an unrelated
    "yes" to something else entirely."""
    history = history or []

    pending = _pending_orders.get(user_id)
    if pending is not None:
        confirmation = _parse_confirmation(message)
        if confirmation == "yes":
            reply = _execute_pending_order(user_id)
            return {"reply": reply, "intent": INTENT_PLACE_ORDER, "ticker": _pending_order_ticker(pending)}
        if confirmation == "no":
            _pending_orders.pop(user_id, None)
            return {"reply": "Cancelled -- no order placed.", "intent": INTENT_PLACE_ORDER, "ticker": None}
        _pending_orders.pop(user_id, None)

    classification = classify_intent(message, history)
    intent = classification["intent"]
    ticker = classification["ticker"]

    if intent == INTENT_DAILY_BRIEFING:
        # Special-cased ahead of the normal _HANDLERS dispatch -- see
        # _handle_daily_briefing's own docstring for why its ticker
        # can't come from classify_intent's pre-pass the way every
        # other intent's does.
        briefing = _handle_daily_briefing(user_id)
        return {"reply": briefing["text"], "intent": intent, "ticker": briefing["ticker"]}

    reply = _HANDLERS[intent](user_id, ticker, message, history)

    return {"reply": reply, "intent": intent, "ticker": ticker}
