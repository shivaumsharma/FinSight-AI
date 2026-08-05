"""
rlvr_reward.py

The verifiable reward for GRPO training: a thin wrapper around
scripts/phase2_backtest.py's own score_rating(), so "correct" means
exactly what it already means everywhere else in this project (Buy if
realized_return_pct > +5%, Sell if < -5%, Hold in between -- see
phase2_backtest.BUY_THRESHOLD/SELL_THRESHOLD) rather than a
newly-invented definition.

Deliberately binary (1.0 / 0.0), not shaped by return magnitude or a
separate format-compliance bonus -- a first RLVR pass should prove the
simple version works before adding reward-shaping complexity (see the
project's plan doc). A completion that doesn't follow the requested
"Verdict: <Buy/Hold/Sell>" format scores 0.0, same as a wrong verdict --
an unusable answer isn't worth more than a wrong one.
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from phase2_backtest import score_rating  # noqa: E402

from app.training.rlvr_prompt import parse_verdict


def compute_reward(completion: str, realized_return_pct: float) -> float:
    """Single-example reward, 1.0 or 0.0. Used directly by unit tests
    and by anything scoring one rollout at a time."""
    verdict = parse_verdict(completion)
    if verdict is None:
        return 0.0
    correct = score_rating(verdict, realized_return_pct)
    if correct is None:
        # realized_return_pct missing/None -- shouldn't happen for data
        # produced by rlvr_context.build_point_in_time_context, which
        # always returns a real float, but score_rating's own contract
        # allows None, so honor it the same way phase2_backtest.py does
        # (unscored, not wrong).
        return 0.0
    return 1.0 if correct else 0.0


def grpo_reward_fn(completions: List[str], realized_return_pct: List[float], **kwargs) -> List[float]:
    """trl.GRPOTrainer-compatible reward function: takes the batch of
    sampled completions plus the `realized_return_pct` dataset column
    trl passes through automatically (each training example carries its
    own ground-truth realized return, produced once by
    scripts/build_rlvr_dataset.py -- never recomputed at train time),
    and returns one float reward per completion, same order."""
    return [
        compute_reward(completion, realized_return)
        for completion, realized_return in zip(completions, realized_return_pct)
    ]


def verdict_from_completion(completion: str) -> Optional[str]:
    """Re-exported for callers (evaluate_rlvr_model.py) that need the
    parsed verdict itself, not just the reward -- avoids importing
    rlvr_prompt directly in places that only care about scoring."""
    return parse_verdict(completion)
