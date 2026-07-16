"""
comparison_tool.py

Implements the "Compare Apple and Microsoft" branch of the
architecture:

    Planner -> run MarketData/Valuation/(RAG) for both companies
            -> ComparisonTool
            -> ReportTool

ComparisonTool builds a second ResearchContext for the peer company
(reusing MarketDataTool and ValuationTool exactly as-is), then writes
a synthesized `research_summary` covering both companies onto the
*primary* context so the existing ReportTool/PromptBuilder/
ReportGenerator can produce a single comparative report with no
changes to any of that code.

The peer ticker is resolved once by the ResearchAgent (via
company_resolver.resolve_companies) and passed in through
context.metadata["peer_ticker"] -- ComparisonTool itself does no
entity extraction.
"""

from app.core.research_context import ResearchContext
from .base_tool import BaseTool
from .market_data_tool import MarketDataTool
from .valuation_tool import ValuationTool
from .rag_tool import RAGTool


class ComparisonTool(BaseTool):

    name = "comparison_tool"
    description = (
        "Runs market data and valuation for a second ('peer') company and "
        "builds a side-by-side comparison summary. Only used when the "
        "question compares two companies (e.g. 'Compare Apple and Microsoft')."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        peer_ticker = context.metadata.get("peer_ticker")

        if not peer_ticker or peer_ticker.upper() == context.ticker.upper():
            context.record_tool(self.name)
            return context

        peer_context = ResearchContext(
            ticker=peer_ticker,
            question=context.question,
        )

        MarketDataTool().run(peer_context)
        ValuationTool().run(peer_context)
        RAGTool().run(peer_context)

        context.metadata["peer_financial_summary"] = peer_context.financial_summary
        context.metadata["peer_valuation_summary"] = peer_context.valuation_summary
        context.metadata["peer_company_info"] = peer_context.company_info

        context.research_summary = self._build_comparison_summary(context, peer_context)

        context.record_tool(self.name)

        return context

    # -----------------------------------------------------------

    def _build_comparison_summary(self, primary: ResearchContext, peer: ResearchContext) -> str:

        def block(ctx: ResearchContext) -> str:
            info = ctx.company_info or {}
            lines = [
                f"Company: {info.get('company_name', ctx.ticker)} ({ctx.ticker})",
                f"Sector: {info.get('sector', 'Unknown')}",
            ]
            for key, value in (ctx.financial_summary or {}).items():
                lines.append(f"  {key}: {value}")
            for key, value in (ctx.valuation_summary or {}).items():
                lines.append(f"  {key}: {value}")
            return "\n".join(lines)

        return (
            "=" * 60
            + f"\nCOMPARISON: {primary.ticker} vs {peer.ticker}\n"
            + "=" * 60
            + f"\n\n--- {primary.ticker} ---\n"
            + block(primary)
            + f"\n\n--- {peer.ticker} ---\n"
            + block(peer)
        )
