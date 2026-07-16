"""
institutional_consensus_tool.py

Computes the Institutional Consensus Score (ICS): how closely
FinSight's own recommendation agrees with real institutional analyst
ratings for the same ticker. This is an evaluation/market-context
metric only -- see app/reporting/consensus_score.py's module
docstring for the full reasoning on why it can never influence
FinSight's own recommendation, valuation, or thesis.

Deliberately re-derives FinSight's rating via the same pure
derive_recommendation() function report_data_builder.py itself calls,
rather than reading it back from an already-built report -- this tool
has no code path that writes into FinSight's recommendation, even by
accident, since it never touches report_tool's output at all.
"""

from app.core.research_context import ResearchContext
from app.reporting.report_data_builder import derive_recommendation
from app.reporting.institutional_ratings import fetch_institutional_ratings
from app.reporting.consensus_score import build_consensus_report
from .base_tool import BaseTool


class InstitutionalConsensusTool(BaseTool):

    name = "institutional_consensus_tool"
    description = (
        "Compares FinSight's own Buy/Hold/Sell recommendation against real "
        "institutional analyst ratings for the ticker, purely as a market-context "
        "metric. Never changes FinSight's recommendation, valuation, or thesis. "
        "Needs valuation_tool to have already run."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        if not context.valuation_results:
            context.record_tool(self.name)
            return context

        finsight_rating = derive_recommendation(context.valuation_results)["rating"]

        institutional_ratings = fetch_institutional_ratings(context.ticker)

        # Nested under "recommendation_consensus" (not stored flat) so
        # future versions can add sibling dimensions -- valuation,
        # thesis, risk, sentiment comparisons -- without reshaping
        # what this version already produces.
        context.institutional_consensus = {
            "recommendation_consensus": build_consensus_report(
                finsight_rating, institutional_ratings
            ),
            "valuation_consensus": None,
            "thesis_consensus": None,
            "risk_consensus": None,
            "sentiment_consensus": None,
        }

        context.record_tool(self.name)

        return context
