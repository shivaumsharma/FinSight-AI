"""
rag_tool.py

Retrieves grounded evidence from an earnings-call transcript.

Reuses, unchanged:
 - RAGPipeline            (ingestion + vector query)
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
        "Retrieves and reranks relevant earnings-call transcript chunks for the "
        "user's question, with citations. Required for questions about "
        "management commentary, earnings calls, guidance, or transcripts. "
        "Needs context.transcript_path to be set."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        if not context.transcript_path:
            context.record_tool(self.name)
            return context

        pipeline = RAGPipeline()

        context.transcript_chunks = pipeline.ingest_transcript(
            context.transcript_path
        )

        intent = QueryClassifier().classify(context.question)
        context.add_metadata("query_intent", intent)

        retrieval_query = RetrievalPlanner().build(context.question, intent)

        chunks = pipeline.query_pipeline(retrieval_query, n_results=20)

        # Operator remarks ("welcome to the call...") are boilerplate
        # and never carry investable information.
        chunks = [
            chunk
            for chunk in chunks
            if chunk.get("metadata", {}).get("speaker", "").lower() != "operator"
        ]

        context.retrieved_chunks = EvidenceReranker().rerank(
            context.question, chunks, top_k=5
        )

        context.citations = CitationEngine().build(context.retrieved_chunks)

        context.record_tool(self.name)

        return context
