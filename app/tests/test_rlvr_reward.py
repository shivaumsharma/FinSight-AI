"""
Unit tests for app/training/rlvr_reward.py and rlvr_prompt.parse_verdict.

No network, no model -- pure string parsing and arithmetic against
scripts/phase2_backtest.py's own score_rating(), so these run in
milliseconds and belong in CI same as the other deterministic-math test
files in this directory.
"""

from app.training.rlvr_prompt import parse_verdict
from app.training.rlvr_reward import compute_reward, grpo_reward_fn


def test_parse_verdict_extracts_buy():
    assert parse_verdict("Some reasoning here.\nVerdict: Buy") == "Buy"


def test_parse_verdict_is_case_insensitive():
    assert parse_verdict("Verdict: sell") == "Sell"


def test_parse_verdict_returns_none_with_no_verdict_line():
    assert parse_verdict("The company looks solid but I won't commit to a call.") is None


def test_parse_verdict_uses_last_match_when_model_second_guesses():
    text = "Initially this looks like a Buy, but on reflection...\nVerdict: Hold"
    assert parse_verdict(text) == "Hold"


def test_reward_is_one_for_correct_buy():
    # phase2_backtest.BUY_THRESHOLD is +5.0 -- 12% clears it.
    assert compute_reward("reasoning\nVerdict: Buy", realized_return_pct=12.0) == 1.0


def test_reward_is_zero_for_incorrect_buy():
    # A Buy call where the stock actually fell is wrong.
    assert compute_reward("reasoning\nVerdict: Buy", realized_return_pct=-8.0) == 0.0


def test_reward_is_one_for_correct_sell():
    assert compute_reward("reasoning\nVerdict: Sell", realized_return_pct=-9.0) == 1.0


def test_reward_is_one_for_correct_hold():
    # SELL_THRESHOLD=-5.0, BUY_THRESHOLD=+5.0 -- 1% sits inside the band.
    assert compute_reward("reasoning\nVerdict: Hold", realized_return_pct=1.0) == 1.0


def test_reward_is_zero_for_hold_outside_band():
    assert compute_reward("reasoning\nVerdict: Hold", realized_return_pct=9.0) == 0.0


def test_reward_is_zero_for_unparseable_completion():
    # A right-sounding but wrongly-formatted answer scores the same as
    # a wrong one -- an unusable answer isn't worth more (see this
    # module's own docstring).
    assert compute_reward("I'd say this is a strong buy.", realized_return_pct=50.0) == 0.0


def test_grpo_reward_fn_scores_a_batch_in_order():
    completions = [
        "Verdict: Buy",   # correct: +10% clears +5%
        "Verdict: Sell",  # incorrect: stock actually rose
        "no verdict at all",  # unparseable
    ]
    realized = [10.0, 20.0, 0.0]
    assert grpo_reward_fn(completions, realized) == [1.0, 0.0, 0.0]
