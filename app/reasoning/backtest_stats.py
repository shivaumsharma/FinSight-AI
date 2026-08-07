"""
backtest_stats.py

Real, measured accuracy of this app's own Buy/Hold/Sell recommendation
logic, computed from scripts/phase2_backtest.py's saved results
(scripts/backtest_results_curated_asof{12,24}mo_*.json) -- NOT a
per-ticker or per-category number (most researched tickers aren't in
the curated 79-ticker backtest universe, so there's no honest way to
compute one), but the same aggregate number report_data_builder.py's
own comments already treat as this system's headline accuracy stat.

Two windows exist because the backtest was run twice, against two
different historical cohorts:
- "asof12mo_exit0mo": as-of 12 months ago, exit today -- the more
  recent cohort.
- "asof24mo_exit12mo": as-of 24 months ago, exit 12 months ago -- an
  older, out-of-sample cohort.
Both score the exact same thing (a Buy/Hold/Sell call against its
actual 12-month forward return, using phase2_backtest.py's own
+/-5% BUY_THRESHOLD/SELL_THRESHOLD bands) -- the "correct" field on
each row is that scoring already precomputed when the file was
generated, so this module just aggregates it rather than re-deriving
it. The recent-cohort window is the "primary" number shown as the
badge; the older cohort is exposed as "secondary" context, not hidden,
since the plain 52.6%-vs-32.9% gap between the two is real trained-model-
adjacent honesty this app should not paper over.
"""

import json
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
_PRIMARY_FILE = BASE_DIR / "scripts" / "backtest_results_curated_asof12mo_exit0mo.json"
_SECONDARY_FILE = BASE_DIR / "scripts" / "backtest_results_curated_asof24mo_exit12mo.json"

_PRIMARY_WINDOW_LABEL = "12-month forward return, curated universe"
_SECONDARY_WINDOW_LABEL = "12-month forward return, older cohort (24-month lookback)"


def _load_window_stats(path: Path, window_label: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError):
        # Missing/corrupt backtest file must degrade to "no badge shown",
        # not take down the report endpoint that renders it.
        return None
    scored = [r for r in records if r.get("correct") is not None]
    if not scored:
        return None
    correct = sum(1 for r in scored if r["correct"])
    return {
        "accuracy_pct": round(correct / len(scored) * 100, 1),
        "correct": correct,
        "scored": len(scored),
        "window_label": window_label,
    }


_primary_cache = _load_window_stats(_PRIMARY_FILE, _PRIMARY_WINDOW_LABEL)
_secondary_cache = _load_window_stats(_SECONDARY_FILE, _SECONDARY_WINDOW_LABEL)


def get_backtest_accuracy_summary() -> Optional[dict]:
    """{"accuracy_pct", "correct", "scored", "window_label",
    "secondary": {...} | None} for the badge every recommendation card
    shows, or None if the primary window's backtest file is missing/
    unreadable (the badge is then simply not rendered, never a
    fabricated placeholder number)."""
    if _primary_cache is None:
        return None
    return {**_primary_cache, "secondary": _secondary_cache}
