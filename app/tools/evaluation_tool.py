"""
evaluation_tool.py

Wraps the existing EvaluationEngine (grounding, retrieval, citation
coverage, completeness, latency) as a tool so it shows up in the
registry/trace like every other step, instead of being called ad hoc
from streamlit_app.py as it was before.

Latency is derived from `context.request_time`, which
`ResearchContext` already stamps at construction time via
`datetime.utcnow()` -- no new bookkeeping needed.
"""

from datetime import datetime

from app.core.research_context import ResearchContext
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

        context.record_tool(self.name)

        return context
