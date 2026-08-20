"""
Smoke test for scripts/evaluate_rlvr_model.py -- this script had never
been run successfully before it was actually needed: a stale
`_naive_baseline_accuracy` import (the real name is
`naive_baseline_accuracy`, no leading underscore) meant it crashed
with an ImportError the very first time anyone tried to use it. This
test only guards against that exact class of regression (the module
failing to import at all) -- the real generation logic is exercised
live against a real model in scripts/diagnose_rlvr_generation.py and
by actually running this script, not re-mocked here.
"""

import importlib
import sys
from pathlib import Path


def test_module_imports_without_error():
    scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    # A fresh import (not relying on any prior cached import elsewhere
    # in the test session) is what actually would have caught the
    # stale-name ImportError before it cost a live run.
    sys.modules.pop("evaluate_rlvr_model", None)
    importlib.import_module("evaluate_rlvr_model")


def test_naive_baseline_accuracy_name_matches_phase2_backtest():
    scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import phase2_backtest

    # The exact regression: an import of a name with a leading
    # underscore that was never actually exported under that name.
    assert hasattr(phase2_backtest, "naive_baseline_accuracy")
    assert not hasattr(phase2_backtest, "_naive_baseline_accuracy")


def _load():
    scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    sys.modules.pop("evaluate_rlvr_model", None)
    return importlib.import_module("evaluate_rlvr_model")


def test_true_bucket_matches_score_ratings_own_thresholds():
    m = _load()
    assert m._true_bucket(m.BUY_THRESHOLD + 0.1) == "Buy"
    assert m._true_bucket(m.SELL_THRESHOLD - 0.1) == "Sell"
    assert m._true_bucket(0.0) == "Hold"
    assert m._true_bucket(m.BUY_THRESHOLD) == "Hold"  # boundary is inclusive on the Hold side
    assert m._true_bucket(m.SELL_THRESHOLD) == "Hold"


def test_confusion_matrix_does_not_crash_on_real_rows(capsys):
    m = _load()
    rows = [
        {"realized_return_pct": 20.0, "recommendation": "Buy"},   # true Buy, predicted Buy
        {"realized_return_pct": -20.0, "recommendation": "Buy"},  # true Sell, predicted Buy (wrong)
        {"realized_return_pct": 0.0, "recommendation": "Hold"},   # true Hold, predicted Hold
    ]
    m._print_confusion_matrix(rows)
    output = capsys.readouterr().err
    assert "Confusion matrix" in output
    assert "Per-class precision/recall/F1" in output


def test_confusion_matrix_handles_empty_input_without_raising():
    m = _load()
    m._print_confusion_matrix([])  # must not raise on zero scoreable rows
