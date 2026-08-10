"""
rag_tool.py

Retrieves grounded evidence from a real regulatory disclosure --
SEC EDGAR (an 8-K earnings-release exhibit, or a 10-Q/10-K's MD&A
section as fallback) for US-listed tickers, NSE India (a "Financial
Result Updates"/"Outcome of Board Meeting" announcement) for .NS-listed
ones -- fetched live for whatever ticker the question is about, not
from a handful of hand-authored transcript files for a fixed list of
companies. See RAGPipeline.ingest_company_disclosure for the SEC-vs-NSE
branch; everything below that point is source-agnostic.

Reuses, unchanged:
 - QueryClassifier         (rule-based intent detection)
 - RetrievalPlanner        (expands the query per intent)
 - CitationEngine          (builds citation objects)

This is the same sequence EvidenceBuilder used to run inline; it is
now an independently invocable tool so the planner can call RAG on
its own (e.g. "summarize the earnings call") without also paying for
a DCF valuation.

EvidenceReranker (app/rag/reranker.py, a general-purpose cross-encoder,
cross-encoder/ms-marco-MiniLM-L-6-v2) is intentionally NOT used here
anymore. Measured with a real hand-labeled eval (scripts/
evaluate_retrieval.py + app/evaluation/retrieval_labels.py, 7 real
queries across 5 companies) plus a live A/B report-generation test: the
reranker made retrieval WORSE on 6 of 7 queries (mean Precision@5
0.552 -> 0.438, MRR 0.786 -> 0.548), and in the live test, citation
grounding score for the same NFLX question went from 0.0 (with
reranker) to 60.0 (raw retrieval, no reranker) -- with the reranker
active, the generated narrative wasn't actually grounded in any of
the evidence it was given. A cross-encoder trained on general web
search relevance (MS MARCO) apparently doesn't transfer well to dense
financial filing text. Raw embedding retrieval (BAAI/bge-base-en-v1.5)
is used directly instead -- also faster, one fewer model in memory.
"""

from app.core.research_context import ResearchContext
from app.rag.rag_pipeline import RAGPipeline
from app.reasoning.query_classifier import QueryClassifier
from app.reasoning.retrieval_planner import RetrievalPlanner
from app.rag.citation_engine import CitationEngine
from .base_tool import BaseTool

# Matches EvidenceReranker's prior top_k=5 default -- unrelated to
# n_results=20 below, which is how many candidates come back from raw
# vector search before trimming to this many for the narrative prompt.
TOP_K_EVIDENCE = 5


class RAGTool(BaseTool):

    name = "rag_tool"
    description = (
        "Retrieves relevant evidence from the company's most recent regulatory "
        "disclosure (earnings release or filing), with citations. Required "
        "for questions about management commentary, guidance, or recent results. "
        "Works for any ticker SEC (US-listed) or NSE (.NS-listed) has a filing for."
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

        # Raw embedding retrieval, no reranking step -- see module
        # docstring for why. ChromaVectorStore.query_documents already
        # returns results ordered by embedding similarity, so querying
        # for exactly TOP_K_EVIDENCE results directly (rather than
        # over-fetching 20 and trimming) is both simpler and correct
        # now that there's no reranker to give the extra candidates to.
        context.retrieved_chunks = pipeline.query_pipeline(
            retrieval_query, ticker=context.ticker, n_results=TOP_K_EVIDENCE
        )

        context.citations = CitationEngine().build(context.retrieved_chunks, disclosure)

        context.record_tool(self.name)

        return context
