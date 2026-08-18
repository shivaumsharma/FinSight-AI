"""
rating_alerts.py

Feature 2: notices when a stock a user owns or watches gets a new
rating from a fresh research report, and proactively proposes a
response instead of waiting to be asked. Runs as an on-demand sweep
(see main.py's POST /v1/internal/sweep/rating-changes), same
cron-triggered pattern as app/reasoning/call_tracker.py's
check_matured_checkpoints -- not an in-process background thread (see
main.py's own module docstring for why: Cloud Run billing).

The proposal itself is just a plain assistant chat_messages row (with
`ticker` set for carry-over, see chat_router.py's _most_recent_ticker)
-- replying to it ("yes, sell it") flows through the Foundation's
existing place_order intent and confirm-then-execute flow. No separate
execution path is built here.
"""

import logging

from app.api import auth, db
from app.reasoning.chat_router import INTENT_PLACE_ORDER

logger = logging.getLogger(__name__)


def _tracked_tickers(user_id: str) -> set:
    """Union of a user's watchlist + portfolio tickers -- "everything
    this user is paying attention to," the same scope Feature 5's daily
    briefing reuses this function for."""
    watchlist = {row["ticker"] for row in db.get_watchlist(user_id)}
    portfolio = {row["ticker"] for row in db.get_portfolio_holdings(user_id)}
    return watchlist | portfolio


def _suggested_action(rating: str) -> str:
    if rating == "Sell":
        return "exit your position"
    if rating == "Buy":
        return "add to your position"
    return "review it"


def _notify_rating_change(user_id: str, ticker: str, rating: str) -> None:
    """Writes the proposal chat message, then best-effort push-notifies
    every subscription. The push half is wrapped in its own try/except
    (never propagates) -- same non-fatal side-effect discipline
    jobs.py's _notify_job_complete already uses: a dead/unreachable push
    subscription must never stop the proposal itself from having been
    written, or (via the caller's own try/except) from being marked
    seen."""
    body = f"{ticker} just moved to {rating} -- want me to {_suggested_action(rating)}?"
    db.add_chat_message(user_id, "assistant", body, intent=INTENT_PLACE_ORDER, ticker=ticker)

    try:
        for sub in db.get_push_subscriptions(user_id):
            try:
                auth.send_push_notification(sub, f"{ticker} rating change", body, "info")
            except Exception as e:
                logger.warning(f"[rating-change sweep] push failed for one subscription (non-fatal): {e}")
    except Exception as e:
        logger.warning(f"[rating-change sweep] push lookup failed for user={user_id} (non-fatal): {e}")


def sweep_rating_changes() -> int:
    """For every user, every tracked ticker whose latest completed
    rating differs from what they were last notified about (see
    db.get_rating_alert_seen -- None the first time a ticker is ever
    seen, so its current rating always counts as worth an initial
    proposal) gets exactly one proposal message, and
    db.set_rating_alert_seen() is updated in the SAME pass so the next
    sweep sees it as already-seen and won't re-fire until the rating
    genuinely changes again. Returns the number of proposals sent.

    A single user/ticker's proposal failing (a write error, a push
    lookup blowing up despite its own internal try/except) is caught
    and logged, never allowed to stop the sweep from reaching every
    other user -- same per-item isolation GET /v1/watchlist already
    uses for a bad quote. Crucially, set_rating_alert_seen is only
    called on SUCCESS -- a failed proposal must not be marked seen, or
    the user would silently never be told about a real rating change.
    """
    sent = 0
    for user_id in db.get_all_user_ids():
        for ticker in _tracked_tickers(user_id):
            rating = db.get_latest_rating_for_ticker(user_id, ticker)
            if rating is None:
                continue  # never researched -- nothing to compare

            if rating == db.get_rating_alert_seen(user_id, ticker):
                continue  # no change since the last notification

            try:
                _notify_rating_change(user_id, ticker, rating)
            except Exception as e:
                logger.warning(f"[rating-change sweep] failed for user={user_id} ticker={ticker} (non-fatal): {e}")
                continue

            db.set_rating_alert_seen(user_id, ticker, rating)
            sent += 1

    return sent
