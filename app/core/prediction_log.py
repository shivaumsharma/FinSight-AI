"""
prediction_log.py

Lightweight, append-only log of every recommendation this system has
ever made, kept separate from ResearchLogger's full per-report JSON
dumps (app/core/logger.py) -- those are heavy (whole normalized
financials, full generated report text, etc.) and written one file
per run, which is fine for auditing a single report but awkward to
scan across hundreds of runs. This is the opposite: one line per
report, only the handful of fields an accuracy check (Phase 2-style,
but on live, non-backtested predictions) actually needs.

Wired into ReportTool.run() (see app/tools/report_tool.py), not a client --
ReportTool is the terminal tool of "almost every plan" (its own docstring),
so every real report gets logged here regardless of which entry point
produced it (API, a script), not just whichever caller remembers to call
ResearchLogger.save().
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.reporting.report_data_builder import compute_signal_agreement

DEFAULT_LOG_PATH = os.path.join("logs", "prediction_log.jsonl")

# The local file is lost on every redeploy. When Snowflake is configured (SNOWFLAKE_* env vars, see
# scripts/snowflake_accuracy_store.py) each entry is also written to this table; without it, nothing changes.
SNOWFLAKE_TABLE = "PREDICTION_LOG"
_SNOWFLAKE_COLUMNS = ["timestamp", "ticker", "recommendation", "price_at_call", "upside_percent", "challenger_verdict",
                      "data_quality_level", "range_low_1q", "range_high_1q", "agreement", "grounding_score", "overall_score"]
_CREATE_SQL = f"""CREATE TABLE IF NOT EXISTS {SNOWFLAKE_TABLE} (
    timestamp TIMESTAMP_TZ, ticker STRING, recommendation STRING, price_at_call FLOAT, upside_percent FLOAT,
    challenger_verdict STRING, data_quality_level STRING, range_low_1q FLOAT, range_high_1q FLOAT,
    agreement STRING, grounding_score FLOAT, overall_score FLOAT)"""


class PredictionLogger:

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def log(self, ticker: str, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Appends one line to the JSONL log from an already-built
        report_data dict (report_data_builder.build_report_data's
        return value) and returns the entry that was written, so
        callers/tests can verify without re-reading the file.

        Called from EvaluationTool (not ReportTool) specifically so
        confidence_scores -- grounding/overall -- are already the
        real, computed values rather than the "Unavailable"
        placeholder report_tool bakes in before evaluation has run.
        Kept here (not investigated/fixed as part of the grounding-
        score drift found this session) so grounding accumulates
        across enough real reports to assess the trend on a larger
        sample instead of guessing from a handful of runs.
        """
        recommendation = report_data.get("recommendation", {})
        valuation = report_data.get("valuation_analysis", {})
        confidence = report_data.get("confidence_scores", {})
        relative_valuation = valuation.get("relative_valuation")
        rating = recommendation.get("rating")

        risk_range = (report_data.get("risk_range") or {}).get("ranges", {}).get("1 quarter") or {}
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "recommendation": rating,
            # Price at the call, so the outcome can be scored later; the challenger is the display-only ML
            # classifier's verdict, logged so it can be compared with the rating on the same live calls.
            "price_at_call": (report_data.get("market_earnings_snapshot") or {}).get("current_price"),
            "challenger_verdict": (valuation.get("ml_classifier") or {}).get("verdict"),
            "data_quality_level": (report_data.get("data_quality") or {}).get("level"),
            "range_low_1q": risk_range.get("low"),
            "range_high_1q": risk_range.get("high"),
            "dcf_available": valuation.get("DCF Available"),
            "upside_percent": self._none_if_unavailable(valuation.get("Upside (%)")),
            "relative_valuation_signal": (relative_valuation or {}).get("signal"),
            "agreement": compute_signal_agreement(rating, relative_valuation),
            "grounding_score": self._none_if_unavailable(confidence.get("Grounding (%)")),
            "overall_score": self._none_if_unavailable(confidence.get("Overall Score")),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        threading.Thread(target=self._mirror_to_snowflake, args=(entry,), daemon=True).start()
        return entry

    @staticmethod
    def _mirror_to_snowflake(entry: Dict[str, Any]) -> bool:
        """Best-effort durable copy; False (never an exception) when not configured or unreachable."""
        try:
            from scripts import snowflake_accuracy_store as store

            conn = store.connect()
            if conn is None:
                return False
            try:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_SQL)
                    cur.execute(
                        f"INSERT INTO {SNOWFLAKE_TABLE} ({', '.join(_SNOWFLAKE_COLUMNS)}) "
                        f"VALUES ({', '.join(['%s'] * len(_SNOWFLAKE_COLUMNS))})",
                        [entry.get(c) for c in _SNOWFLAKE_COLUMNS],
                    )
                return True
            finally:
                conn.close()
        except Exception:
            return False

    @staticmethod
    def _none_if_unavailable(value):
        return None if value == "Unavailable" else value

    def read_all(self):
        """Returns every logged entry, oldest first. Used for the
        eventual walk-forward accuracy check, and for tests/proof."""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
