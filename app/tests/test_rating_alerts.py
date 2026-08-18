"""
Unit tests for app/reasoning/rating_alerts.py (Feature 2). Real sqlite
temp DB (same pattern as test_push_notifications.py) so
get_latest_rating_for_ticker/get_rating_alert_seen/set_rating_alert_seen
all exercise real transactional behavior; auth.send_push_notification
is monkeypatched, never a real push service call.
"""

import pytest

from app.api import auth, db
from app.reasoning import rating_alerts as ra


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


@pytest.fixture
def sent_notifications(monkeypatch):
    calls = []

    def fake_send(subscription, title, body, status):
        calls.append((subscription["endpoint"], title, body, status))

    monkeypatch.setattr(auth, "send_push_notification", fake_send)
    return calls


def _rate(temp_db, user_id, ticker, rating):
    """Completes a research job for `ticker` with `rating`, so
    get_latest_rating_for_ticker(user_id, ticker) returns it -- the
    real denormalization path (mark_done), not a monkeypatched stub."""
    job_id = temp_db.create_job(question=f"research {ticker}", orchestrator="hand_rolled", user_id=user_id)
    temp_db.mark_running(job_id)
    temp_db.mark_done(job_id, {"ticker": ticker, "report_data": {"recommendation": {"rating": rating}}}, None)


# ---------------------------------------------------------------- _tracked_tickers

def test_tracked_tickers_unions_watchlist_and_portfolio(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "AAPL")
    temp_db.upsert_portfolio_holding(user_id, "MSFT", 5, 100.0)

    assert ra._tracked_tickers(user_id) == {"AAPL", "MSFT"}


def test_tracked_tickers_deduplicates_overlap(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "AAPL")
    temp_db.upsert_portfolio_holding(user_id, "AAPL", 5, 100.0)

    assert ra._tracked_tickers(user_id) == {"AAPL"}


# ---------------------------------------------------------------- sweep_rating_changes

def test_sweep_sends_exactly_one_proposal_per_genuine_change(temp_db, sent_notifications):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh-val", "auth-val")
    temp_db.add_watchlist_item(user_id, "TCS")
    _rate(temp_db, user_id, "TCS", "Hold")

    sent_first = ra.sweep_rating_changes()
    assert sent_first == 1
    assert len(sent_notifications) == 1
    assert "TCS just moved to Hold" in sent_notifications[0][2]

    # Same run again with no rating change -- must NOT re-fire.
    sent_second = ra.sweep_rating_changes()
    assert sent_second == 0
    assert len(sent_notifications) == 1


def test_sweep_fires_again_only_after_a_genuine_flip(temp_db, sent_notifications):
    # The spec's own acceptance check: flip a rating between two sweep
    # runs -- exactly one new proposal, and it stops re-firing again
    # until the rating changes once more.
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh-val", "auth-val")
    temp_db.add_watchlist_item(user_id, "TCS")
    _rate(temp_db, user_id, "TCS", "Hold")

    ra.sweep_rating_changes()
    assert len(sent_notifications) == 1

    _rate(temp_db, user_id, "TCS", "Sell")  # the flip
    sent = ra.sweep_rating_changes()
    assert sent == 1
    assert len(sent_notifications) == 2
    assert "TCS just moved to Sell" in sent_notifications[1][2]
    assert "exit your position" in sent_notifications[1][2]

    # Unchanged since the flip -- disappears from future sweeps again.
    ra.sweep_rating_changes()
    assert len(sent_notifications) == 2


def test_sweep_skips_a_ticker_with_no_completed_rating(temp_db, sent_notifications):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "TCS")  # never researched

    sent = ra.sweep_rating_changes()

    assert sent == 0
    assert sent_notifications == []


def test_sweep_covers_portfolio_holdings_too(temp_db, sent_notifications):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_push_subscription(user_id, "https://push.example.com/ep1", "p256dh-val", "auth-val")
    temp_db.upsert_portfolio_holding(user_id, "AAPL", 4, 150.0)
    _rate(temp_db, user_id, "AAPL", "Buy")

    sent = ra.sweep_rating_changes()

    assert sent == 1
    assert "add to your position" in sent_notifications[0][2]


def test_sweep_writes_an_assistant_chat_message_with_ticker_for_carry_over(temp_db):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "TCS")
    _rate(temp_db, user_id, "TCS", "Sell")

    ra.sweep_rating_changes()

    messages = temp_db.list_chat_messages(user_id)
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert messages[0]["ticker"] == "TCS"
    assert "TCS just moved to Sell" in messages[0]["content"]


def test_sweep_is_isolated_per_user(temp_db, sent_notifications):
    u1 = temp_db.create_user("a@example.com", "h", "s")
    u2 = temp_db.create_user("b@example.com", "h", "s")
    temp_db.add_watchlist_item(u1, "TCS")
    _rate(temp_db, u1, "TCS", "Sell")

    ra.sweep_rating_changes()

    assert temp_db.list_chat_messages(u2) == []


def test_sweep_survives_a_broken_push_subscription_lookup(temp_db, monkeypatch):
    # A push failure must never stop the proposal from being written or
    # from being marked seen -- same non-fatal discipline as
    # jobs.py's _notify_job_complete.
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "TCS")
    _rate(temp_db, user_id, "TCS", "Sell")
    monkeypatch.setattr(db, "get_push_subscriptions", lambda user_id: (_ for _ in ()).throw(RuntimeError("db blip")))

    sent = ra.sweep_rating_changes()  # must not raise

    assert sent == 1
    assert db.get_rating_alert_seen(user_id, "TCS") == "Sell"


def test_sweep_does_not_mark_seen_when_the_proposal_write_itself_fails(temp_db, monkeypatch):
    user_id = temp_db.create_user("a@example.com", "h", "s")
    temp_db.add_watchlist_item(user_id, "TCS")
    _rate(temp_db, user_id, "TCS", "Sell")
    monkeypatch.setattr(db, "add_chat_message", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("disk full")))

    sent = ra.sweep_rating_changes()  # must not raise, and must not count a failed send

    assert sent == 0
    # Not marked seen -- a genuinely failed proposal must be retried on
    # the next sweep, not silently treated as delivered.
    assert db.get_rating_alert_seen(user_id, "TCS") is None
