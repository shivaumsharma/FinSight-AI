"""
baseline_scoring.py

The Buy/Hold/Sell scoring rule and naive-baseline accuracy calculation
-- extracted verbatim from scripts/phase2_backtest.py (the one-off
historical backtest harness) so the live call-outcome tracker
(app/reasoning/call_tracker.py) scores real, ongoing calls against the
EXACT same rule the historical backtest already validated, not a
second copy that could quietly drift from it. phase2_backtest.py now
imports these from here instead of defining them locally.

The accuracy definition itself (Buy correct if realized return
> BUY_THRESHOLD, Sell if < SELL_THRESHOLD, Hold if between) is
untouched -- moving it here is a location change, not a redefinition.
"""

BUY_THRESHOLD = 5.0
SELL_THRESHOLD = -5.0


def score_rating(rating, realized_return_pct):
    """Scores an arbitrary rating string against a realized return.
    None for "Insufficient Data"/None rating or a missing return --
    never a fabricated correct/incorrect on unscoreable input."""
    if rating is None or rating == "Insufficient Data" or realized_return_pct is None:
        return None
    if rating == "Buy":
        return realized_return_pct > BUY_THRESHOLD
    if rating == "Sell":
        return realized_return_pct < SELL_THRESHOLD
    if rating == "Hold":
        return SELL_THRESHOLD <= realized_return_pct <= BUY_THRESHOLD
    return None


def naive_baseline_accuracy(scored_rows, fixed_rating):
    """Accuracy of a rater that ALWAYS says `fixed_rating`, scored the
    same way the real model is -- a Buy/Hold/Sell prediction's accuracy
    is nothing more than "how often did the market agree" if the model
    itself doesn't beat this. `scored_rows` is any iterable of dicts
    with a "realized_return_pct" key -- the live scoreboard passes
    scored tracked_call_checkpoints rows, phase2_backtest.py passes its
    own historical rows; both are scored identically."""
    scores = [score_rating(fixed_rating, r["realized_return_pct"]) for r in scored_rows]
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return None
    return 100 * sum(valid_scores) / len(valid_scores)
