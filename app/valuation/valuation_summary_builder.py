"""
valuation_summary_builder.py

Summarizes DCF valuation results.
"""

from app.core.research_context import ResearchContext


class ValuationSummaryBuilder:

    def build(self, context: ResearchContext) -> ResearchContext:

        valuation = context.valuation_results or {}

        summary = f"""
Intrinsic Value:
{valuation.get("intrinsic_value","Unavailable")}

Enterprise Value:
{context.enterprise_value}

Equity Value:
{context.equity_value}

Market Cap:
{context.market_cap}
"""

        context.valuation_summary = summary

        return context