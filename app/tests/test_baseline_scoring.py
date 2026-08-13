"""
Unit tests for app/analysis/baseline_scoring.py -- pure functions, no
mocking needed. Moved from scripts/phase2_backtest.py; these tests
also stand in as the regression check that the extraction didn't
change the accuracy definition.
"""

from app.analysis import baseline_scoring as bs


# ---------------------------------------------------------------- score_rating

def test_buy_correct_above_threshold():
    assert bs.score_rating("Buy", 6.0) is True


def test_buy_incorrect_below_threshold():
    assert bs.score_rating("Buy", 4.0) is False


def test_buy_incorrect_exactly_at_threshold():
    assert bs.score_rating("Buy", 5.0) is False


def test_sell_correct_below_threshold():
    assert bs.score_rating("Sell", -6.0) is True


def test_sell_incorrect_above_threshold():
    assert bs.score_rating("Sell", -4.0) is False


def test_hold_correct_inside_band():
    assert bs.score_rating("Hold", 0.0) is True
    assert bs.score_rating("Hold", 5.0) is True
    assert bs.score_rating("Hold", -5.0) is True


def test_hold_incorrect_outside_band():
    assert bs.score_rating("Hold", 5.1) is False
    assert bs.score_rating("Hold", -5.1) is False


def test_insufficient_data_is_never_scored():
    assert bs.score_rating("Insufficient Data", 10.0) is None


def test_none_rating_is_never_scored():
    assert bs.score_rating(None, 10.0) is None


def test_missing_realized_return_is_never_scored():
    assert bs.score_rating("Buy", None) is None


def test_unknown_rating_string_is_never_scored():
    assert bs.score_rating("Strong Buy", 10.0) is None


# ---------------------------------------------------------------- naive_baseline_accuracy

def test_naive_baseline_accuracy_always_buy():
    rows = [{"realized_return_pct": 10.0}, {"realized_return_pct": -10.0}, {"realized_return_pct": 0.0}]
    # Always-Buy: correct on the +10 row only (Buy needs >5.0) -> 1/3
    assert bs.naive_baseline_accuracy(rows, "Buy") == 100 / 3


def test_naive_baseline_accuracy_always_hold():
    rows = [{"realized_return_pct": 0.0}, {"realized_return_pct": -10.0}, {"realized_return_pct": 10.0}]
    # Always-Hold: correct on the 0 row only -> 1/3
    assert bs.naive_baseline_accuracy(rows, "Hold") == 100 / 3


def test_naive_baseline_accuracy_skips_missing_returns():
    rows = [{"realized_return_pct": 10.0}, {"realized_return_pct": None}]
    assert bs.naive_baseline_accuracy(rows, "Buy") == 100.0


def test_naive_baseline_accuracy_is_none_for_empty_input():
    assert bs.naive_baseline_accuracy([], "Buy") is None


def test_naive_baseline_accuracy_is_none_when_every_row_is_unscoreable():
    rows = [{"realized_return_pct": None}, {"realized_return_pct": None}]
    assert bs.naive_baseline_accuracy(rows, "Buy") is None
