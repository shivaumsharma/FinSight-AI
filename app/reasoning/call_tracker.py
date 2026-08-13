"""
call_tracker.py

The live "does the model beat doing nothing" scoreboard -- logs every
real Buy/Hold/Sell call a report makes, checks back on it at fixed
7/30/90-day checkpoints, and scores it against the exact same
Buy/Hold/Sell rule and naive-baseline comparison already proven out in
scripts/phase2_backtest.py's one-off historical backtest (see
app/analysis/baseline_scoring.py -- the shared module both now import
from, so this can never quietly drift from what the backtest already
validated).

No-look-ahead discipline, same spirit as phase2_backtest.py's own
module docstring: check_matured_checkpoints() prices each checkpoint
using the historical close ON (nearest to) its own checkpoint_date --
never "whatever the price happens to be right now when the sweep
thread wakes up," which would silently use a later, unrelated price if
the sweep runs late (the same class of bug this feature exists to
guard against, per the DCF-engine look-ahead incident this task was
scoped from).
"""

import time
from datetime import datetime, timezone
from typing import Optional

from app.analysis.baseline_scoring import naive_baseline_accuracy, score_rating
from app.api import db
from app.data.market_data import MarketDataLoader


def log_tracked_call(job_id: Optional[str], result: dict) -> Optional[str]:
    """Called right after db.mark_done() persists a completed report.
    Skips logging entirely for a None/"Insufficient Data" rating --
    there's nothing to track a naive baseline against. Same
    .get()-chain-never-raises convention db.mark_done() already uses
    against this exact `result` shape -- a malformed/partial result
    must not break the report it's attached to just because tracking
    couldn't extract a field."""
    report_data = result.get("report_data") or {}
    rating = (report_data.get("recommendation") or {}).get("rating")
    if not rating or rating == "Insufficient Data":
        return None

    ticker = result.get("ticker")
    price_at_call = (report_data.get("market_earnings_snapshot") or {}).get("current_price")
    if not ticker or price_at_call is None:
        return None

    currency = report_data.get("currency") or "USD"
    confidence = (report_data.get("signal_quality") or {}).get("confidence")

    return db.insert_tracked_call(
        job_id=job_id, ticker=ticker, rating=rating, confidence=confidence,
        currency=currency, price_at_call=price_at_call, called_at=time.time(),
    )


def _closest_price_on_or_near(historical_prices, checkpoint_date: float) -> Optional[float]:
    """The historical close nearest checkpoint_date (a unix timestamp)
    -- not necessarily an exact match, since weekends/holidays mean the
    checkpoint date itself is often not a trading day. Nearest by
    calendar distance in either direction: being a day early or late
    is immaterial noise, not a systematic bias (unlike using today's
    price for a checkpoint that was due days ago, which IS a bias)."""
    if historical_prices is None or historical_prices.empty:
        return None
    checkpoint_dt = datetime.fromtimestamp(checkpoint_date, tz=timezone.utc).date()
    closes = historical_prices["Close"]
    diffs = [(abs((idx.date() - checkpoint_dt).days), idx) for idx in closes.index]
    diffs.sort(key=lambda pair: pair[0])
    closest_idx = diffs[0][1]
    return float(closes.loc[closest_idx])


def check_matured_checkpoints() -> int:
    """Scores every checkpoint whose due date has arrived and hasn't
    been scored yet. Returns the count scored this run (0 is a normal,
    frequent result -- most sweep runs have nothing due). Never raises:
    one ticker's historical-price fetch failing must not stop the rest
    of the batch from being scored, same per-item isolation this app
    uses everywhere else (watchlist quotes, corporate actions)."""
    due = db.list_maturable_checkpoints(time.time())
    scored_count = 0

    for row in due:
        try:
            historical_prices = MarketDataLoader(row["ticker"]).get_historical_prices()
            price_at_checkpoint = _closest_price_on_or_near(historical_prices, row["checkpoint_date"])
        except Exception:
            price_at_checkpoint = None

        if price_at_checkpoint is None:
            continue

        price_at_call = row["price_at_call"]
        realized_return_pct = (price_at_checkpoint - price_at_call) / price_at_call * 100

        model_correct = score_rating(row["rating"], realized_return_pct)
        always_buy_correct = score_rating("Buy", realized_return_pct)
        always_hold_correct = score_rating("Hold", realized_return_pct)

        db.record_checkpoint_outcome(
            call_id=row["call_id"], window_days=row["window_days"],
            price_at_checkpoint=price_at_checkpoint, realized_return_pct=realized_return_pct,
            model_correct=model_correct, always_buy_correct=always_buy_correct,
            always_hold_correct=always_hold_correct, scored_at=time.time(),
        )
        scored_count += 1

    return scored_count


_RATING_KEYS = ("Buy", "Sell", "Hold")


def _breakdown(scored_rows: list) -> dict:
    """Per-rating precision: of the calls that said Buy (etc.), how
    many were actually scored correct. Uses model_correct (already
    computed via score_rating against the CALL's own rating), not a
    fresh scoring pass -- this is a restatement of the same scored
    field, split by rating, not a second judgment."""
    breakdown = {}
    for key in _RATING_KEYS:
        rows_for_rating = [r for r in scored_rows if r["rating"] == key]
        total = len(rows_for_rating)
        correct = sum(1 for r in rows_for_rating if r["model_correct"])
        breakdown[key.lower()] = {
            "correct": correct, "total": total,
            "precision_pct": round(100 * correct / total, 1) if total else None,
        }
    return breakdown


def _window_stats(window_days: int) -> dict:
    scored_rows = db.list_scored_checkpoints(window_days)
    pending_count = db.count_pending_checkpoints(window_days)
    sample_size = len(scored_rows)

    if sample_size == 0:
        return {
            "model_accuracy_pct": None, "sample_size": 0, "pending_count": pending_count,
            "always_buy_accuracy_pct": None, "always_hold_accuracy_pct": None,
            "breakdown": _breakdown([]),
        }

    model_scores = [r["model_correct"] for r in scored_rows if r["model_correct"] is not None]
    model_accuracy_pct = round(100 * sum(model_scores) / len(model_scores), 1) if model_scores else None

    always_buy_accuracy_pct = naive_baseline_accuracy(scored_rows, "Buy")
    always_hold_accuracy_pct = naive_baseline_accuracy(scored_rows, "Hold")
    if always_buy_accuracy_pct is not None:
        always_buy_accuracy_pct = round(always_buy_accuracy_pct, 1)
    if always_hold_accuracy_pct is not None:
        always_hold_accuracy_pct = round(always_hold_accuracy_pct, 1)

    return {
        "model_accuracy_pct": model_accuracy_pct, "sample_size": sample_size, "pending_count": pending_count,
        "always_buy_accuracy_pct": always_buy_accuracy_pct, "always_hold_accuracy_pct": always_hold_accuracy_pct,
        "breakdown": _breakdown(scored_rows),
    }


def get_scoreboard() -> dict:
    """{"windows": {"7": {...}, "30": {...}, "90": {...}}, "generated_at": float}
    -- see _window_stats for the per-window shape. Every stat is None
    (never a fabricated 0) when a window has zero scored checkpoints,
    so the frontend can tell "no data yet" apart from "a real 0%"."""
    return {
        "windows": {str(w): _window_stats(w) for w in db.TRACKED_WINDOWS_DAYS},
        "generated_at": time.time(),
    }
