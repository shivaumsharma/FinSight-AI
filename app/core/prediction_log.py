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

Wired into ReportTool.run() (see app/tools/report_tool.py), not
Streamlit -- ReportTool is the terminal tool of "almost every plan"
(its own docstring), so every real report gets logged here regardless
of which entry point produced it (Streamlit UI, a script, a future
API), not just whichever caller remembers to call ResearchLogger.save().
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.reporting.report_data_builder import compute_signal_agreement

DEFAULT_LOG_PATH = os.path.join("logs", "prediction_log.jsonl")


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

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "recommendation": rating,
            "dcf_available": valuation.get("DCF Available"),
            "upside_percent": self._none_if_unavailable(valuation.get("Upside (%)")),
            "relative_valuation_signal": (relative_valuation or {}).get("signal"),
            "agreement": compute_signal_agreement(rating, relative_valuation),
            "grounding_score": self._none_if_unavailable(confidence.get("Grounding (%)")),
            "overall_score": self._none_if_unavailable(confidence.get("Overall Score")),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

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
