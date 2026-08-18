"""
daily_briefing.py

Feature 5: one short, spoken-friendly portfolio digest -- P&L, rating
changes since the last briefing (reusing Feature 2's own
rating_alerts_seen table via db.get_rating_alerts_seen_since), and
upcoming corporate actions for held/watched tickers (reusing
app/reporting/corporate_actions_feed.py, the same body GET
/v1/corporate-actions/feed already uses). A summary VIEW over data
these other features already produce, not a new decision engine -- if
it names a rating change, the reply flows through the same
place_order/Foundation confirm-and-execute path every other proposal
in this app already uses (ticker carry-over via the assistant
chat_messages row, see chat_router.py's _most_recent_ticker).

build_briefing() is pure/read-only (no side effects, so it's cheap to
call from either a chat turn or a sweep) -- advancing
db.last_briefing_at is each CALLER's own job (chat_router.py's
on-demand handler, and sweep_daily_briefings below), since only they
know whether the briefing was actually successfully delivered to the
user (a chat reply always reaches them; a sweep-sent one depends on the
chat-message write and push notification below).

Same "raw grounding data doubles as both the LLM's context AND the
LLM-outage fallback" pattern chat_router.py's _handle_portfolio_status/
_handle_ticker_question already established -- a briefing is still a
real, honest answer even when the LLM backend is down, just a flatter
one.
"""

import logging
import time

from app.api import auth, db
from app.core.llm_provider import HostedProvider, LLMProviderError
from app.reasoning.chat_router import CHAT_MODEL, DISCLAIMER_NOTE, INTENT_DAILY_BRIEFING, PERSONA
from app.reporting.corporate_actions_feed import build_corporate_actions_feed
from app.reporting.portfolio_summary import build_portfolio_view

logger = logging.getLogger(__name__)

# A short digest, not the full feed GET /v1/corporate-actions/feed
# already exposes elsewhere -- only the soonest few events are worth
# naming out loud in a briefing.
MAX_BRIEFING_EVENTS = 3

# A user is only sweep-briefed once per calendar-ish day -- 20 hours
# (not a strict 24) gives cron-timing drift some slack without letting
# two sweep-triggered briefings land on the same real day. Also the
# gate against a user who just asked on-demand minutes ago getting
# immediately re-briefed by the next sweep run (see
# db.set_last_briefing_at's own docstring: both paths share one cursor).
DAILY_BRIEFING_MIN_INTERVAL_SECONDS = 20 * 3600


def _portfolio_line(user_id: str) -> str:
    view = build_portfolio_view(user_id)
    holdings = view["holdings"]
    if not holdings:
        return "No holdings in the portfolio yet."

    total_value = view["summary"].get("total_market_value")
    if total_value is None:
        return f"{len(holdings)} holding(s), but current value couldn't be priced right now."

    pnl = view["summary"].get("total_unrealized_pnl")
    pnl_pct = view["summary"].get("total_unrealized_pnl_pct")
    if pnl is None:
        return f"Portfolio: {len(holdings)} holding(s), ${total_value:,.2f} total (USD equiv.)."

    direction = "up" if pnl >= 0 else "down"
    return (
        f"Portfolio: {len(holdings)} holding(s), ${total_value:,.2f} total (USD equiv.), "
        f"{direction} ${abs(pnl):,.2f} ({pnl_pct:+.1f}%) overall."
    )


def _event_lines(user_id: str) -> list:
    feed = build_corporate_actions_feed(user_id, scope="all")
    lines = []
    for e in feed["events"][:MAX_BRIEFING_EVENTS]:
        label = "earnings" if e["type"] == "earnings" else "ex-dividend"
        lines.append(f"{e['ticker']} {label} on {e['date']}")
    return lines


def build_briefing(user_id: str) -> dict:
    """{"text": str, "ticker": str | None} -- ticker is set ONLY when
    exactly one rating change is being mentioned, so a follow-up like
    "yes, exit it" carries over unambiguously (the same single-ticker
    carry-over every other proposal in this app relies on); None for
    zero or multiple changes, since there's no single "it" to resolve
    to. Never raises -- same LLM-outage-degrades-to-raw-text discipline
    as every other synthesis handler in chat_router.py."""
    user = db.get_user_by_id(user_id)
    since = (user.get("last_briefing_at") if user else None) or 0

    portfolio_line = _portfolio_line(user_id)
    rating_changes = db.get_rating_alerts_seen_since(user_id, since)
    rating_lines = [f"{c['ticker']} moved to {c['last_notified_rating']}" for c in rating_changes]
    event_lines = _event_lines(user_id)

    lines = [portfolio_line]
    if rating_lines:
        lines.append("Rating changes since your last briefing: " + "; ".join(rating_lines) + ".")
    if event_lines:
        lines.append("Coming up: " + "; ".join(event_lines) + ".")
    raw_text = "\n".join(lines)

    ticker = rating_changes[0]["ticker"] if len(rating_changes) == 1 else None

    prompt = (
        f"{PERSONA} Compose the daily portfolio briefing data below into ONE short, spoken-friendly paragraph "
        "(2-4 sentences, no bullet points or headers -- this may be read aloud) -- use ONLY the real data given, "
        "never invent numbers. If a rating change is mentioned, end with a specific, actionable callout offering "
        "to act on it (e.g. \"want me to exit/add to it?\"), not a vague \"let me know if you need anything\". "
        f"{DISCLAIMER_NOTE}\n\nBRIEFING DATA:\n{raw_text}"
    )
    try:
        provider = HostedProvider(model=CHAT_MODEL)
        text = provider.generate(prompt, max_new_tokens=200).strip()
    except LLMProviderError:
        text = raw_text

    return {"text": text, "ticker": ticker}


def sweep_daily_briefings() -> int:
    """For every user whose last briefing (on-demand or sweep-sent
    alike) is more than DAILY_BRIEFING_MIN_INTERVAL_SECONDS old, builds
    and delivers one: writes it as a plain assistant chat message
    (ticker set for carry-over) and best-effort push-notifies every
    subscription -- same non-fatal push discipline as
    rating_alerts.py/price_alerts.py. A single user's briefing failing
    is caught and logged, never allowed to stop the sweep from reaching
    everyone else; last_briefing_at is only advanced on a SUCCESSFUL
    delivery (same "don't mark it seen if the write itself failed"
    discipline as those two sibling sweeps), so a failed user gets
    retried on the next run rather than silently skipped for a day.
    Returns the number of briefings sent."""
    sent = 0
    now = time.time()
    for user_id in db.get_all_user_ids():
        user = db.get_user_by_id(user_id)
        last = user.get("last_briefing_at") if user else None
        if last is not None and now - last < DAILY_BRIEFING_MIN_INTERVAL_SECONDS:
            continue

        try:
            briefing = build_briefing(user_id)
            db.add_chat_message(user_id, "assistant", briefing["text"], intent=INTENT_DAILY_BRIEFING, ticker=briefing["ticker"])
        except Exception as e:
            logger.warning(f"[daily-briefing sweep] failed for user={user_id} (non-fatal): {e}")
            continue

        try:
            for sub in db.get_push_subscriptions(user_id):
                try:
                    auth.send_push_notification(sub, "Your daily briefing is ready", briefing["text"][:150], "info")
                except Exception as e:
                    logger.warning(f"[daily-briefing sweep] push failed for one subscription (non-fatal): {e}")
        except Exception as e:
            logger.warning(f"[daily-briefing sweep] push lookup failed for user={user_id} (non-fatal): {e}")

        db.set_last_briefing_at(user_id, now)
        sent += 1

    return sent
