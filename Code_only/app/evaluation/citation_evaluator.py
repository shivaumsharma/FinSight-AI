"""
citation_evaluator.py

Measures evidence utilization.
"""

from dataclasses import dataclass


@dataclass
class CitationResult:

    citations_available: int
    citations_used: int
    citation_coverage: float


class CitationEvaluator:

    def evaluate(
        self,
        citations,
        generated_report: str,
    ) -> CitationResult:

        used = 0

        report = generated_report.lower()

        for citation in citations:

            text = citation["text"][:40].lower()

            if text in report:
                used += 1

        total = len(citations)

        coverage = (

            used / total * 100

            if total

            else 0

        )

        return CitationResult(

            citations_available=total,

            citations_used=used,

            citation_coverage=round(coverage, 2),

        )