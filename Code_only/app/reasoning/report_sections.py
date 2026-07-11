"""
report_sections.py

Generates each section of the investment report and
combines them into a final report.
"""

from app.core.prompt_builder import PromptBuilder
from app.reasoning.report_composer import ReportComposer


class ReportSectionGenerator:

    def __init__(self, llm):

        self.llm = llm
    # =====================================================
    # Main Report Generation
    # =====================================================

    def generate(self, context):

        builder = PromptBuilder()

        summary = self.generate_summary(
            builder.build_summary_prompt(context)
        )

        bull = self.generate_bull_case(
            builder.build_bull_prompt(context)
        )

        bear = self.generate_bear_case(
            builder.build_bear_prompt(context)
        )

        financial = self.generate_financial_outlook(
            builder.build_financial_prompt(context)
        )

        recommendation = self.generate_recommendation(
            builder.build_recommendation_prompt(context)
        )

        composer = ReportComposer()

        return composer.compose(
            summary,
            recommendation
        )