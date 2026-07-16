"""
report_tool.py

Terminal tool of almost every plan. Consumes whatever the earlier
tools populated on the context (financials, valuation, sentiment,
RAG evidence -- any subset, including none) and produces the final
institutional-style equity research report.

The report is a hybrid, not one LLM blob:
 - Numeric/structured sections (Company Overview, Financial Statement
   Analysis, Ratio Analysis, Growth Analysis, Valuation Analysis,
   Confidence Scores, References, the Buy/Hold/Sell rating) are built
   directly from already-computed data -- see report_data_builder.py.
   These are always accurate; no LLM involved.
 - Narrative sections (Executive Summary, Business Analysis, Market
   and Earnings Analysis, Risk Analysis, Investment Thesis) are
   written by the local LLM in a single combined call -- see
   narrative_builder.py -- and must stay consistent with the
   deterministic recommendation rather than declaring their own.

context.generated_answer is still populated (the narrative sections
concatenated) since the evaluation engine's grounding/citation/
completeness checks operate on that string. context.report_data holds
everything needed to render the downloadable PDF.
"""

from app.core.research_context import ResearchContext
from app.analysis.report_builder import ResearchSummaryBuilder
from app.reporting.report_data_builder import build_report_data
from app.reporting.narrative_builder import build_narrative_sections, NARRATIVE_SECTIONS
from app.reporting.pdf_report_builder import build_pdf_report
from .base_tool import BaseTool


class ReportTool(BaseTool):

    name = "report_tool"
    description = (
        "Synthesizes all previously gathered evidence (financials, valuation, "
        "sentiment, transcript citations) into a final institutional-style "
        "equity research report. Should almost always be the last tool in "
        "any plan."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        context.research_summary = ResearchSummaryBuilder().build(context)

        report_data = build_report_data(context)

        narrative = build_narrative_sections(context, report_data)
        report_data["narrative"] = narrative

        context.report_data = report_data

        context.generated_answer = "\n\n".join(
            f"# {section}\n{narrative[section]}" for section in NARRATIVE_SECTIONS
        )

        context.pdf_bytes = build_pdf_report(report_data)

        context.record_tool(self.name)

        return context
