from dataclasses import dataclass


@dataclass
class EvaluationMetrics:

    latency: float

    answer_length: int

    grounding_score: float
    supported_claims: int
    unsupported_claims: int

    retrieval_score: float
    retrieved_chunks: int
    reranked_chunks: int

    citation_score: float
    citations_used: int
    citations_available: int

    # Same mechanism as citation_score, scoped to news articles and
    # "[News N]" tags instead of SEC evidence -- see
    # citation_evaluator.py's evaluate_news(). 0 with news_available=0
    # means "no news coverage to ground against", not "ungrounded".
    news_grounding_rate: float
    news_citations_used: int
    news_citations_available: int

    completeness_score: float
    missing_sections: list

    overall_score: float