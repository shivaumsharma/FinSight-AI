"""
evaluation_engine.py

Evaluates the quality of the generated response.

This module DOES NOT generate answers.

It only measures the quality of the pipeline output
and stores the results inside the ResearchContext.
"""

import time

from app.core.research_context import ResearchContext


class EvaluationEngine:

    def __init__(self):
        pass

    def evaluate(
        self,
        context: ResearchContext,
        start_time: float,
        end_time: float
    ) -> ResearchContext:

        metrics = {}

        # ========================================
        # Latency
        # ========================================

        metrics["latency"] = round(
            end_time - start_time,
            3
        )

        # ========================================
        # Retrieval Count
        # ========================================

        metrics["context.retrieved_chunks"] = len(
            context.retrieved_chunks
        )

        # ========================================
        # Evidence Used
        # ========================================

        metrics["evidence_available"] = (
            len(context.retrieved_chunks) > 0
        )

        # ========================================
        # Response Length
        # ========================================

        metrics["response_length"] = len(
            context.generated_answer.split()
        ) if context.generated_answer else 0

        # ========================================
        # Valuation Available
        # ========================================

        metrics["valuation_generated"] = (
            bool(context.valuation_results)
        )

        # ========================================
        # Sentiment Available
        # ========================================

        metrics["sentiment_generated"] = (
            bool(context.sentiment)
        )

        # ========================================
        # Placeholder Metrics
        # (Future)
        # ========================================

        metrics["precision"] = None

        metrics["recall"] = None

        metrics["faithfulness"] = None

        metrics["answer_relevancy"] = None

        # Save into context

        context.evaluation= metrics

        return context