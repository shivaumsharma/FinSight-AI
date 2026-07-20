"""
evaluation_tool.py

Wraps the existing EvaluationEngine (grounding, retrieval, citation
coverage, completeness, latency) as a tool so it shows up in the
registry/trace like every other step, instead of being called ad hoc
from streamlit_app.py as it was before.

Latency is derived from `context.request_time`, which
`ResearchContext` already stamps at construction time via
`datetime.utcnow()` -- no new bookkeeping needed.

Confidence Scores is one of the report's own 13 sections, but the
scores it shows can only be known after the report has already been
written and evaluated -- report_tool builds context.report_data (with
those scores) before this tool ever runs, so it necessarily bakes in
placeholder "Unavailable" values. This tool patches the real scores
into context.report_data and re-renders the PDF afterward, rather than
leaving the report showing scores it never actually had.
"""

from datetime import datetime

from app.core.research_context import ResearchContext
from app.core.prediction_log import PredictionLogger
from app.evaluation.evaluation_engine import EvaluationEngine
from .base_tool import BaseTool


class EvaluationTool(BaseTool):

    name = "evaluation_tool"
    description = (
        "Scores the generated report for grounding, retrieval quality, citation "
        "coverage, and structural completeness. Always runs last, after "
        "report_tool, regardless of the planner's chosen tool list."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        if not context.generated_answer:
            context.record_tool(self.name)
            return context

        latency = (datetime.utcnow() - context.request_time).total_seconds()

        metrics = EvaluationEngine().evaluate(
            context=context,
            generated_report=context.generated_answer,
            latency=latency,
        )

        context.evaluation = vars(metrics)

        self._refresh_report(context)

        # Forward-tracking log (prediction_log.py): written here, not
        # from report_tool, specifically so grounding_score/
        # overall_score are actually populated -- this tool is the
        # first point in the plan where they exist. Never allowed to
        # break the run -- a logging failure (e.g. a read-only
        # filesystem) shouldn't cost the user their report.
        try:
            if context.report_data:
                PredictionLogger().log(context.ticker, context.report_data)
        except Exception:
            pass

        context.record_tool(self.name)

        return context

    def _refresh_report(self, context: ResearchContext) -> None:
        if not context.report_data:
            return

        evaluation = context.evaluation
        context.report_data["confidence_scores"] = {
            "Overall Score": evaluation.get("overall_score", "Unavailable"),
            "Grounding (%)": evaluation.get("grounding_score", "Unavailable"),
            "Retrieval (%)": evaluation.get("retrieval_score", "Unavailable"),
            "Citation Coverage (%)": evaluation.get("citation_score", "Unavailable"),
            "News Grounding Rate (%)": (
                evaluation.get("news_grounding_rate", "Unavailable")
                if context.news_selected
                else "No news coverage available"
            ),
            "Completeness (%)": evaluation.get("completeness_score", "Unavailable"),
        }

        from app.reporting.pdf_report_builder import build_pdf_report
        context.pdf_bytes = build_pdf_report(context.report_data)
