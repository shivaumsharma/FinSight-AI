"""
Unit tests for app/reasoning/call_tracker.py. Uses a real temp SQLite
DB (monkeypatched db.DB_PATH) rather than mocking db.py itself -- this
module is mostly a thin orchestration layer over real db.py functions,
so exercising them for real is more honest than mocking every call.
MarketDataLoader is monkeypatched for check_matured_checkpoints, since
that's the one real network dependency.
"""

import time

import pandas as pd
import pytest

from app.api import db
from app.reasoning import call_tracker as ct


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


def _fake_result(rating="Buy", current_price=100.0, currency="USD", confidence=70.0, ticker="AAPL"):
    return {
        "ticker": ticker,
        "report_data": {
            "recommendation": {"rating": rating},
            "signal_quality": {"confidence": confidence},
            "market_earnings_snapshot": {"current_price": current_price},
            "currency": currency,
        },
    }


# ---------------------------------------------------------------- log_tracked_call

def test_log_tracked_call_writes_a_call_and_three_checkpoints():
    call_id = ct.log_tracked_call("job1", _fake_result())
    assert call_id is not None

    due = db.list_maturable_checkpoints(time.time() + 200 * 86400)  # far enough out that all 3 are "due"
    windows = {row["window_days"] for row in due if row["call_id"] == call_id}
    assert windows == {7, 30, 90}


def test_log_tracked_call_skips_insufficient_data():
    assert ct.log_tracked_call("job1", _fake_result(rating="Insufficient Data")) is None


def test_log_tracked_call_skips_none_rating():
    result = _fake_result()
    result["report_data"]["recommendation"]["rating"] = None
    assert ct.log_tracked_call("job1", result) is None


def test_log_tracked_call_skips_missing_price():
    result = _fake_result()
    result["report_data"]["market_earnings_snapshot"]["current_price"] = None
    assert ct.log_tracked_call("job1", result) is None


def test_log_tracked_call_extracts_confidence_and_currency():
    call_id = ct.log_tracked_call("job1", _fake_result(confidence=82.5, currency="INR"))
    with db._connect() as conn:
        conn.row_factory = None
        row = conn.execute("SELECT confidence, currency FROM tracked_calls WHERE call_id=?", (call_id,)).fetchone()
    assert row == (82.5, "INR")


def test_log_tracked_call_survives_a_malformed_result():
    """Same never-raise convention db.mark_done() itself uses against
    this exact result shape -- a report missing report_data entirely
    must not raise, just skip tracking."""
    assert ct.log_tracked_call("job1", {"ticker": "AAPL"}) is None
    assert ct.log_tracked_call("job1", {}) is None


# ---------------------------------------------------------------- check_matured_checkpoints

def _fake_history(price, as_of_date):
    index = pd.date_range(end=pd.Timestamp(as_of_date, unit="s", tz="UTC"), periods=5, freq="D")
    return pd.DataFrame({"Close": [price] * 5}, index=index)


class _FakeLoader:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_historical_prices(self):
        return _fake_history(115.0, time.time())


def test_check_matured_checkpoints_scores_a_due_checkpoint(monkeypatch):
    monkeypatch.setattr(ct, "MarketDataLoader", _FakeLoader)
    ct.log_tracked_call("job1", _fake_result(rating="Buy", current_price=100.0))

    # Force the 7-day checkpoint into the past so it's due.
    with db._connect() as conn:
        conn.execute("UPDATE tracked_call_checkpoints SET checkpoint_date=? WHERE window_days=7", (time.time() - 1,))

    scored = ct.check_matured_checkpoints()
    assert scored == 1

    rows = db.list_scored_checkpoints(7)
    assert len(rows) == 1
    assert rows[0]["realized_return_pct"] == pytest.approx(15.0)
    assert rows[0]["model_correct"] == 1  # Buy, +15% > 5% threshold
    assert rows[0]["always_buy_correct"] == 1
    assert rows[0]["always_hold_correct"] == 0


def test_check_matured_checkpoints_leaves_a_not_yet_due_checkpoint_alone(monkeypatch):
    monkeypatch.setattr(ct, "MarketDataLoader", _FakeLoader)
    ct.log_tracked_call("job1", _fake_result())  # called_at=now, so no checkpoint is due yet

    scored = ct.check_matured_checkpoints()

    assert scored == 0
    assert db.list_scored_checkpoints(7) == []


def test_check_matured_checkpoints_uses_checkpoint_date_price_not_current_price(monkeypatch):
    """The core no-look-ahead guarantee: if the sweep runs late, it
    must still price the checkpoint at ITS OWN due date, not whatever
    the price happens to be when the sweep thread wakes up."""
    class _LoaderWithHistoryAroundCheckpoint:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_historical_prices(self):
            # A longer history where the checkpoint-date price (110)
            # differs from the most recent ("today's") price (999) --
            # if the sweep used "today's" price instead of the
            # checkpoint's own date, this test would catch it.
            index = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=10, freq="D")
            closes = [100, 102, 105, 108, 110, 150, 300, 500, 800, 999]
            return pd.DataFrame({"Close": closes}, index=index)

    monkeypatch.setattr(ct, "MarketDataLoader", _LoaderWithHistoryAroundCheckpoint)
    ct.log_tracked_call("job1", _fake_result(current_price=100.0))

    # checkpoint_date lands on the 5th row (close=110), 5 days before "today".
    checkpoint_date = time.time() - 5 * 86400
    with db._connect() as conn:
        conn.execute("UPDATE tracked_call_checkpoints SET checkpoint_date=? WHERE window_days=7", (checkpoint_date,))

    ct.check_matured_checkpoints()

    rows = db.list_scored_checkpoints(7)
    assert rows[0]["realized_return_pct"] == pytest.approx(10.0)  # (110-100)/100*100, NOT (999-100)/100


# ---------------------------------------------------------------- get_scoreboard

def test_get_scoreboard_returns_null_stats_for_an_untracked_window():
    board = ct.get_scoreboard()
    assert board["windows"]["7"]["model_accuracy_pct"] is None
    assert board["windows"]["7"]["sample_size"] == 0
    assert board["windows"]["30"]["model_accuracy_pct"] is None
    assert board["windows"]["90"]["model_accuracy_pct"] is None


def test_get_scoreboard_reports_pending_count_for_unscored_calls(monkeypatch):
    monkeypatch.setattr(ct, "MarketDataLoader", _FakeLoader)
    ct.log_tracked_call("job1", _fake_result())

    board = ct.get_scoreboard()

    assert board["windows"]["7"]["pending_count"] == 1
    assert board["windows"]["7"]["sample_size"] == 0  # not scored yet, so not counted as a sample


def test_get_scoreboard_aggregates_scored_checkpoints_and_breakdown(monkeypatch):
    monkeypatch.setattr(ct, "MarketDataLoader", _FakeLoader)  # always returns close=115.0
    ct.log_tracked_call("job1", _fake_result(rating="Buy", current_price=100.0, ticker="AAPL"))
    ct.log_tracked_call("job2", _fake_result(rating="Sell", current_price=100.0, ticker="MSFT"))

    with db._connect() as conn:
        conn.execute("UPDATE tracked_call_checkpoints SET checkpoint_date=? WHERE window_days=7", (time.time() - 1,))

    ct.check_matured_checkpoints()
    board = ct.get_scoreboard()

    window = board["windows"]["7"]
    assert window["sample_size"] == 2
    assert window["pending_count"] == 0
    # Buy call: +15% return -> correct. Sell call: +15% return -> incorrect (Sell needs <-5%).
    assert window["model_accuracy_pct"] == 50.0
    assert window["breakdown"]["buy"] == {"correct": 1, "total": 1, "precision_pct": 100.0}
    assert window["breakdown"]["sell"] == {"correct": 0, "total": 1, "precision_pct": 0.0}
    assert window["breakdown"]["hold"] == {"correct": 0, "total": 0, "precision_pct": None}
    # Always-Buy: both rows scored as if rated Buy -> both correct (both +15%) -> 100%.
    assert window["always_buy_accuracy_pct"] == 100.0
    # Always-Hold: neither +15% row falls in the Hold band -> 0%.
    assert window["always_hold_accuracy_pct"] == 0.0
