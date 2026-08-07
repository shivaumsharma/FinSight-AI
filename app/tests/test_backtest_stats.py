"""
Tests for app/reasoning/backtest_stats.py's window-accuracy aggregation.

Uses temp JSON files (tmp_path), never the real scripts/*.json files --
this module's own correctness (does it aggregate "correct" fields
right?) is independent of what the real backtest currently says, and
pinning against the real numbers would make this test silently start
failing the next time someone re-runs the backtest with fresh data.
"""

import json

from app.reasoning import backtest_stats


def _write_records(path, records):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)


def test_load_window_stats_computes_accuracy_over_scored_rows_only(tmp_path):
    path = tmp_path / "window.json"
    _write_records(path, [
        {"ticker": "A", "correct": True},
        {"ticker": "B", "correct": True},
        {"ticker": "C", "correct": False},
        {"ticker": "D", "correct": None},  # not scored -- excluded from denominator
    ])
    stats = backtest_stats._load_window_stats(path, "test window")
    assert stats["correct"] == 2
    assert stats["scored"] == 3
    assert stats["accuracy_pct"] == round(2 / 3 * 100, 1)
    assert stats["window_label"] == "test window"


def test_load_window_stats_returns_none_for_missing_file(tmp_path):
    assert backtest_stats._load_window_stats(tmp_path / "nope.json", "x") is None


def test_load_window_stats_returns_none_when_nothing_was_scored(tmp_path):
    path = tmp_path / "window.json"
    _write_records(path, [{"ticker": "A", "correct": None}])
    assert backtest_stats._load_window_stats(path, "x") is None


def test_get_backtest_accuracy_summary_includes_secondary_window(monkeypatch):
    monkeypatch.setattr(backtest_stats, "_primary_cache", {"accuracy_pct": 52.6, "correct": 41, "scored": 78, "window_label": "primary"})
    monkeypatch.setattr(backtest_stats, "_secondary_cache", {"accuracy_pct": 32.9, "correct": 25, "scored": 76, "window_label": "secondary"})

    summary = backtest_stats.get_backtest_accuracy_summary()
    assert summary["accuracy_pct"] == 52.6
    assert summary["secondary"]["accuracy_pct"] == 32.9


def test_get_backtest_accuracy_summary_secondary_is_none_when_that_window_is_missing(monkeypatch):
    monkeypatch.setattr(backtest_stats, "_primary_cache", {"accuracy_pct": 52.6, "correct": 41, "scored": 78, "window_label": "primary"})
    monkeypatch.setattr(backtest_stats, "_secondary_cache", None)

    summary = backtest_stats.get_backtest_accuracy_summary()
    assert summary["secondary"] is None


def test_get_backtest_accuracy_summary_is_none_when_primary_window_is_missing(monkeypatch):
    monkeypatch.setattr(backtest_stats, "_primary_cache", None)
    assert backtest_stats.get_backtest_accuracy_summary() is None


# ---------------------------------------------------------------- deploy-time data dependency

def test_curated_backtest_files_are_tracked_by_git():
    # Regression guard for a real production bug: both curated files
    # were excluded by a blanket "scripts/backtest_results_*.json"
    # gitignore rule meant for regenerated research dumps, so Cloud
    # Build's git clone never had them -- this module's own file reads
    # 404'd live even though the files existed locally the whole time
    # (local tests couldn't have caught it; `git ls-files` is the
    # actual thing that was wrong). A negation rule in .gitignore now
    # un-ignores just these two.
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", str(backtest_stats._PRIMARY_FILE), str(backtest_stats._SECONDARY_FILE)],
        cwd=backtest_stats.BASE_DIR, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert len(tracked) == 2, (
        f"Expected both curated backtest files tracked by git, got: {tracked!r}. "
        "Check .gitignore's negation rules for scripts/backtest_results_curated_*.json."
    )
