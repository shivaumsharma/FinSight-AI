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

    completeness_score: float
    missing_sections: list

    overall_score: float