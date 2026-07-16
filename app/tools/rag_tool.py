"""
rag_tool.py

Retrieves grounded evidence from a real SEC disclosure (an 8-K
earnings-release exhibit, or a 10-Q/10-K's MD&A section as fallback),
fetched live for whatever ticker the question is about -- not from a
handful of hand-authored transcript files for a fixed list of
companies.

Reuses, unchanged:
 - QueryClassifier         (rule-based intent detection)
 - RetrievalPlanner        (expands the query per intent)
 - EvidenceReranker        (cross-encoder reranking)
 - CitationEngine          (builds citation objects)

This is the same sequence EvidenceBuilder used to run inline; it is
now an independently invocable tool so the planner can call RAG on
its own (e.g. "summarize the earnings call") without also paying for
a DCF valuation.
"""

from app.core.research_context import ResearchContext
from app.rag.rag_pipeline import RAGPipeline
from app.reasoning.query_classifier import QueryClassifier
from app.reasoning.retrieval_planner import RetrievalPlanner
from app.rag.reranker import EvidenceReranker
from app.rag.citation_engine import CitationEngine
from .base_tool import BaseTool


class RAGTool(BaseTool):

    name = "rag_tool"
    description = (
        "Retrieves and reranks relevant evidence from the company's most recent "
        "SEC disclosure (earnings release or filing), with citations. Required "
        "for questions about management commentary, guidance, or recent results. "
        "Works for any ticker SEC has a filing for."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        pipeline = RAGPipeline()

        chunks_ingested, disclosure = pipeline.ingest_company_disclosure(context.ticker)

        if not disclosure:
            # SEC has no CIK / no qualifying filing for this ticker --
            # degrade gracefully rather than error, same as a missing
            # transcript used to be handled.
            context.record_tool(self.name)
            return context

        context.add_metadata("disclosure_source", disclosure)
        context.transcript_chunks = chunks_ingested

        intent = QueryClassifier().classify(context.question)
        context.add_metadata("query_intent", intent)

        retrieval_query = RetrievalPlanner().build(context.question, intent)

        chunks = pipeline.query_pipeline(retrieval_query, ticker=context.ticker, n_results=20)

        context.retrieved_chunks = EvidenceReranker().rerank(
            context.question, chunks, top_k=5
        )

        context.citations = CitationEngine().build(context.retrieved_chunks)

        context.record_tool(self.name)

        return context
