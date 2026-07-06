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

from app.core.research_context import ResearchContext

from app.data.market_data import MarketDataLoader
from app.data.financial_normalizer import FinancialStatementNormaliser

from app.rag.rag_pipeline import RAGPipeline

from app.valuation.valuation_pipeline import ValuationPipeline

from app.nlp.finbert import FinBERT


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

        # ============================================
        # RAG Evidence
        # ============================================

        if context.transcript_path:

            context.transcript_chunks = self.rag.ingest_transcript(
                context.transcript_path
            )

            context.retrieved_chunks = self.rag.query_pipeline(
                context.question
            )

        # ============================================
        # Sentiment
        # ============================================

        if context.retrieved_chunks:

            joined_text = "\n".join(context.retrieved_chunks)

            context.sentiment = self.sentiment_model.analyze(
                joined_text
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

        context.enterprise_value = \
            context.valuation_results.get("enterprise_value")

        context.equity_value = \
            context.valuation_results.get("equity_value")

        return context