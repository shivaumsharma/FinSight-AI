"""
accuracy_tearsheet.py

build_accuracy_tearsheet() is GET /v1/accuracy-tearsheet's body,
extracted from main.py the same way build_corporate_actions_feed()/
build_portfolio_view() already are -- see their own module docstrings
for why this project pulls REST endpoint bodies out into app/reporting/
rather than leaving them inline.

Two independent, separately-degrading layers:
  1. The one canonical number (scripts/canonical_accuracy.py's
     canonical_accuracy_result.json) -- the same git-committed artifact
     app/reporting/report_data_builder.py's own _load_track_record()
     reads for the one-line track-record caption shown on every report.
     None if the backtest has never been run/committed.
  2. Live per-sector / Buy-vs-Sell / accuracy-over-time breakdowns from
     Snowflake (scripts/snowflake_accuracy_store.py), when the
     SNOWFLAKE_* env vars are configured. None whenever they aren't, or
     if the query fails for any reason -- this is strictly additive
     detail on top of (1), never a reason to fail the whole tearsheet.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from scripts import snowflake_accuracy_store

_RESULT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "canonical_accuracy_result.json"


def _load_canonical_result() -> Optional[Dict[str, Any]]:
    try:
        with open(_RESULT_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_live_breakdowns() -> Optional[Dict[str, list]]:
    conn = snowflake_accuracy_store.connect()
    if conn is None:
        return None
    try:
        results = snowflake_accuracy_store.run_validation_queries(conn)
        # Row tuples -> lists so this is plain JSON, not something a
        # caller needs to know is "really" a Snowflake row type.
        return {name: [list(row) for row in rows] for name, rows in results.items()}
    except Exception:
        return None
    finally:
        conn.close()


def build_accuracy_tearsheet() -> dict:
    canonical = _load_canonical_result()
    return {
        "available": canonical is not None,
        "canonical": canonical,
        "live_breakdowns": _load_live_breakdowns(),
    }
