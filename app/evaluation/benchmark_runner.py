"""
benchmark_runner.py

Loads a benchmark case (app/benchmarks/*.json) and scores a
generated report against its expected sentiment/recommendation/topics.

The original version imported `ReportScorer` from
`app.evaluation.scorer`, but that module only ever defined
`ScoreAggregator` (used by EvaluationEngine for a completely
different purpose -- aggregating grounding/retrieval/citation/
completeness into one score). `ReportScorer` never existed anywhere
in the codebase, so `BenchmarkRunner().evaluate(...)` raised
ImportError before this fix. A minimal, self-contained
`BenchmarkScore` is defined here instead of resurrecting a class that
was never implemented.
"""

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkScore:

    sentiment_match: bool
    recommendation_match: bool
    topic_coverage: float
    matched_topics: List[str] = field(default_factory=list)
    missing_topics: List[str] = field(default_factory=list)


class BenchmarkRunner:

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, benchmark_path, generated_report: str) -> BenchmarkScore:

        benchmark = self.load(benchmark_path)
        report_lower = generated_report.lower()

        expected_sentiment = benchmark.get("expected_sentiment", "").lower()
        sentiment_match = bool(expected_sentiment) and expected_sentiment in report_lower

        expected_recommendation = benchmark.get("expected_recommendation", "").lower()
        recommendation_match = (
            bool(expected_recommendation) and expected_recommendation in report_lower
        )

        expected_topics = benchmark.get("expected_topics", [])
        matched = [t for t in expected_topics if t.lower() in report_lower]
        missing = [t for t in expected_topics if t not in matched]

        coverage = (
            round(len(matched) / len(expected_topics) * 100, 2)
            if expected_topics
            else 0.0
        )

        return BenchmarkScore(
            sentiment_match=sentiment_match,
            recommendation_match=recommendation_match,
            topic_coverage=coverage,
            matched_topics=matched,
            missing_topics=missing,
        )
