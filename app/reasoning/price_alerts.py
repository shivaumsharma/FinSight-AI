"""
price_alerts.py

Feature 3: a standing "check this in the background" price condition --
"sell if AAPL drops below $180" -- checked by an on-demand sweep (see
main.py's POST /v1/internal/sweep/price-alerts), same cron-triggered
pattern app/reasoning/rating_alerts.py's Feature 2 already established
(see that module's own docstring for why this is an endpoint, not a
background thread).

Non-auto alerts reuse the exact same "write a proposal chat message,
let the reply flow through the Foundation's place_order intent" shape
Feature 2 uses -- no separate execution path. Auto-execute alerts are
the one case in this whole spec where db.execute_order is called
directly from a sweep rather than from a user's own confirm turn --
opting into that is deliberately harder to reach than a plain alert
(see chat_router.py's alert-creation parser for the required,
separately-worded opt-in), and the user is always notified with the
real outcome afterward, whether it filled or couldn't.
"""

import logging

from app.api import auth, db
from app.core.currency import currency_symbol
from app.data.market_data import get_quote
from app.reasoning.chat_router import INTENT_PLACE_ORDER, suggest_quantity

logger = logging.getLogger(__name__)


def _is_triggered(direction: str, price: float, target_price: float) -> bool:
    if direction == "below":
        return price <= target_price
    return price >= target_price  # "above"


def _direction_verb(direction: str) -> str:
    return "dropped below" if direction == "below" else "rose above"


def _notify(user_id: str, ticker: str, title: str, body: str) -> None:
    """Writes the plain assistant chat message, then best-effort
    push-notifies every subscription -- same non-fatal push discipline
    as rating_alerts.py's own _notify_rating_change (a dead/unreachable
    subscription must never stop the chat message itself, or the
    caller's own "mark triggered only on success" step, from
    happening)."""
    db.add_chat_message(user_id, "assistant", body, intent=INTENT_PLACE_ORDER, ticker=ticker)
    try:
        for sub in db.get_push_subscriptions(user_id):
            try:
                auth.send_push_notification(sub, title, body, "info")
            except Exception as e:
                logger.warning(f"[price-alert sweep] push failed for one subscription (non-fatal): {e}")
    except Exception as e:
        logger.warning(f"[price-alert sweep] push lookup failed for user={user_id} (non-fatal): {e}")


def _auto_execute_quantity(user_id: str, ticker: str, side: str, price: float) -> "float | None":
    """How much to trade when an alert with auto_execute=1 fires. A
    SELL closes out whatever the user actually holds -- the natural
    "stop-loss" reading of a SELL price alert; there's no sensible
    smaller amount to pick. A BUY reuses Feature 4's own sizing rule
    (suggest_quantity) rather than inventing a second, different
    formula for the same "how much" question. None (never a fabricated
    fallback) when there's nothing real to size off of -- no holding to
    sell, or an empty portfolio to size a buy against."""
    if side == "SELL":
        holdings = {h["ticker"]: h["quantity"] for h in db.get_portfolio_holdings(user_id)}
        held = holdings.get(ticker, 0.0)
        return held if held > 0 else None

    suggestion = suggest_quantity(user_id, ticker, price)
    return suggestion[0] if suggestion else None


def _fire_auto_execute(alert: dict, price: float, currency: str) -> None:
    user_id, ticker, side = alert["user_id"], alert["ticker"], alert["side"]
    symbol = currency_symbol(currency)
    verb = _direction_verb(alert["direction"])
    condition = f"{ticker} {verb} {symbol}{alert['target_price']:,.2f}"

    quantity = _auto_execute_quantity(user_id, ticker, side, price)
    if quantity is None:
        reason = "nothing held to sell" if side == "SELL" else "no portfolio value to size a buy against"
        body = f"{condition} -- couldn't auto-{side.lower()}: {reason}."
        _notify(user_id, ticker, f"{ticker} alert triggered", body)
        return

    rationale = f"price alert at {symbol}{alert['target_price']:,.2f} triggered"
    try:
        db.execute_order(user_id, ticker, side, quantity, price, currency, rationale=rationale)
        body = f"{condition} -- automatically {side} {quantity:g} shares at {symbol}{price:,.2f} (simulated)."
    except ValueError as e:
        body = f"{condition} -- tried to automatically {side.lower()} but it failed: {e}"

    _notify(user_id, ticker, f"{ticker} alert triggered", body)


def _propose_manual(alert: dict, price: float, currency: str) -> None:
    ticker, side = alert["ticker"], alert["side"]
    symbol = currency_symbol(currency)
    verb = _direction_verb(alert["direction"])
    action = "exit your position" if side == "SELL" else "add to your position"
    body = f"{ticker} {verb} {symbol}{alert['target_price']:,.2f} (now {symbol}{price:,.2f}) -- want me to {action}?"
    _notify(alert["user_id"], ticker, f"{ticker} alert triggered", body)


def sweep_price_alerts() -> int:
    """Checks every active alert's ticker against a live quote; fires
    at most once per alert (db.mark_price_alert_triggered is one-shot,
    see its own docstring) and only after the notification/execution
    for it has actually succeeded -- same "don't mark it seen if the
    write itself failed" discipline as rating_alerts.py. A single
    alert's quote lookup or notification failing is caught and logged,
    never allowed to stop the sweep from checking every other alert.
    Returns the number of alerts fired."""
    fired = 0
    for alert in db.get_active_price_alerts():
        try:
            quote = get_quote(alert["ticker"])
        except Exception as e:
            logger.warning(f"[price-alert sweep] quote failed for {alert['ticker']} (non-fatal): {e}")
            continue

        price = quote["price"]
        if not _is_triggered(alert["direction"], price, alert["target_price"]):
            continue

        currency = quote.get("currency", "USD")
        try:
            if alert["auto_execute"]:
                _fire_auto_execute(alert, price, currency)
            else:
                _propose_manual(alert, price, currency)
        except Exception as e:
            logger.warning(f"[price-alert sweep] failed for alert={alert['alert_id']} (non-fatal): {e}")
            continue

        db.mark_price_alert_triggered(alert["alert_id"])
        fired += 1

    return fired
