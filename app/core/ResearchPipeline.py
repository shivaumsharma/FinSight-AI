"""
evidence_builder.py

Builds a complete ResearchContext by collecting data from
every subsystem.

Responsibilities
----------------
✓ Market Data
✓ Company Information
✓ Financial Statements
✓ Financial Normalization
✓ RAG Retrieval
✓ Sentiment Analysis
✓ Valuation

It performs NO reasoning.

The output of this module becomes the input to the
Reasoning Engine.
"""
from app.reasoning.query_classifier import QueryClassifier
from app.reasoning.retrieval_planner import RetrievalPlanner
from app.core.research_context import ResearchContext
from app.data.market_data import MarketDataLoader
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.rag.reranker import (EvidenceReranker)
from app.rag.rag_pipeline import RAGPipeline
from app.valuation.valuation_summary import(ValuationSummaryBuilder)
from app.nlp.sentiment_summary import SentimentSummaryBuilder
from app.valuation.valuation_pipeline import ValuationPipeline
from app.rag.citation_engine import CitationEngine
from app.nlp.finbert import FinBERT
from app.analysis.report_builder import ResearchSummaryBuilder
from app.analysis.financial_analysis import FinancialAnalysisBuilder

class EvidenceBuilder:

    def __init__(self):

        self.rag = RAGPipeline()

        self.sentiment_model = FinBERT()

    def build(self, context: ResearchContext) -> ResearchContext:

        # ============================================
        # Load Market Data
        # ============================================

        loader = MarketDataLoader(context.ticker)

        context.company_info = loader.get_company_info()

        context.income_statement = loader.get_income_statement()

        context.balance_sheet = loader.get_balance_sheet()

        context.cash_flow = loader.get_cash_flow()

        context.market_cap = context.company_info.get("market_cap")

        context.beta = 1.2

        # ============================================
        # Normalize Financial Statements
        # ============================================

        normaliser = FinancialStatementNormaliser(
            context.income_statement,
            context.balance_sheet,
            context.cash_flow
        )

        context.normalized_financials = normaliser.normalise()

        print("\n========== NORMALIZED FINANCIALS ==========")
        print(type(context.normalized_financials))

        if context.normalized_financials is None:
            print("NORMALIZED DATA IS NONE")
        else:
            print(context.normalized_financials.head())
            print(context.normalized_financials.columns.tolist())
        
        print("\n" + "=" * 80)
        print("NORMALIZED FINANCIALS BEFORE SUMMARY BUILDER")
        print("=" * 80)

        print(type(context.normalized_financials))

        if context.normalized_financials is None:
            print("normalized_financials is None")

        else:
            print("Shape:", context.normalized_financials.shape)
            print("Columns:")
            print(context.normalized_financials.columns.tolist())
            print("\nHead:")
            print(context.normalized_financials.head())

        print("=" * 80 + "\n")
        
        summary_builder = FinancialAnalysisBuilder()

        context.financial_summary=summary_builder.build(context.normalized_financials)
        interpreter=FinancialAnalysisBuilder()

        financial_insights = interpreter.interpret(context.financial_summary)

        context.financial_summary.update(financial_insights)
        # ============================================
        # RAG Evidence
        # ============================================

        if context.transcript_path:

            context.transcript_chunks = self.rag.ingest_transcript(
                context.transcript_path
            )

            classifier = QueryClassifier()

            intent = classifier.classify(
                context.question
            )

            planner = RetrievalPlanner()

            retrieval_query = planner.build(
                context.question,
                intent
            )

            context.add_metadata(
                "query_intent",
                intent
            )

            chunks = self.rag.query_pipeline(
                retrieval_query,
                n_results=20
            )

            # =====================================================
            # Remove low-value transcript chunks
            # =====================================================

            chunks = [

                chunk

                for chunk in chunks

                if chunk.get("metadata",{}).get("speaker","").lower()!= "operator"

            ]
            reranker=EvidenceReranker()

            context.retrieved_chunks = reranker.rerank(
                context.question,
                chunks,
                top_k=5
                        )

            citation_engine = CitationEngine()

            context.citations = citation_engine.build(
                context.retrieved_chunks
            )

        # ============================================
        # Sentiment
        # ============================================

        if context.retrieved_chunks:

            joined_text = "\n".join(chunk["text"]
            for chunk in context.retrieved_chunks)

            context.sentiment = self.sentiment_model.analyze(
                joined_text
            )

            builder = SentimentSummaryBuilder()

            context.sentiment_summary = builder.build(
                context.sentiment
            )

        # ============================================
        # Valuation
        # ============================================

        valuation = ValuationPipeline(
            financial_df=context.normalized_financials,
            market_cap=context.market_cap,
            beta=context.beta
        )

        context.valuation_results = valuation.run_valuation()

        summary_builder = ()

        context.valuation_summary = summary_builder.build(context.valuation_results)
        context.enterprise_value = \
            context.valuation_results.get("enterprise_value")

        context.equity_value = \
            context.valuation_results.get("equity_value")
        context.research_summary= (ResearchSummaryBuilder().build(context))

        return context