"""
evaluation_engine.py

Runs the complete evaluation pipeline.

Responsibilities
----------------
✓ Grounding Evaluation
✓ Retrieval Evaluation
✓ Citation Evaluation
✓ Report Validation
✓ Aggregate Overall Score

Does NOT
---------
✗ Generate reports
✗ Run retrieval
✗ Call the LLM
"""

from app.evaluation.grounding_validator import GroundingValidator
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.citation_evaluator import CitationEvaluator
from app.evaluation.report_validator import ReportValidator
from app.evaluation.scorer import ScoreAggregator

from app.evaluation.metrics import EvaluationMetrics

from app.core.research_context import ResearchContext
print(EvaluationMetrics)
print(EvaluationMetrics.__module__)
print(EvaluationMetrics.__annotations__)

class EvaluationEngine:

    def __init__(self):

        self.grounding_validator = GroundingValidator()

        self.retrieval_evaluator = RetrievalEvaluator()

        self.citation_evaluator = CitationEvaluator()

        self.report_validator = ReportValidator()

        self.score_aggregator = ScoreAggregator()

    def evaluate(
        self,
        context: ResearchContext,
        generated_report: str,
        latency: float,
    ) -> EvaluationMetrics:

        # ============================================
        # Grounding
        # ============================================

        grounding = self.grounding_validator.validate(
            research_summary=context.research_summary,
            generated_report=generated_report,
        )

        # ============================================
        # Retrieval
        # ============================================

        retrieval = self.retrieval_evaluator.evaluate(
            retrieved_chunks=context.transcript_chunks,
            reranked_chunks=context.retrieved_chunks,
        )

        # ============================================
        # Citation Coverage
        # ============================================

        citations = self.citation_evaluator.evaluate(
            citations=context.citations,
            generated_report=generated_report,
        )

        # ============================================
        # Report Validation
        # ============================================

        report = self.report_validator.validate(
            generated_report
        )

        # ============================================
        # Overall Score
        # ============================================

        overall = self.score_aggregator.aggregate(
            grounding=grounding,
            retrieval=retrieval,
            citations=citations,
            report=report,
        )

        # ============================================
        # Return Metrics
        # ============================================

        return EvaluationMetrics(

    latency=round(latency, 2),

    answer_length=len(generated_report),

    grounding_score=grounding.grounding_score,
    supported_claims=grounding.supported_claims,
    unsupported_claims=grounding.unsupported_claims,

    retrieval_score=retrieval.retrieval_score,
    retrieved_chunks=retrieval.retrieved_chunks,
    reranked_chunks=retrieval.reranked_chunks,

    citation_score=citations.citation_coverage,
    citations_used=citations.citations_used,
    citations_available=citations.citations_available,

    completeness_score=report.completeness_score,
    missing_sections=report.missing_sections,

    overall_score=overall,
)