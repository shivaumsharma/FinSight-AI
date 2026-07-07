"""
retrieval_evaluator.py

Evaluates retrieval quality.
"""

from dataclasses import dataclass


@dataclass
class RetrievalResult:

    retrieved_chunks: int
    reranked_chunks: int
    average_chunk_length: float
    retrieval_score: float


class RetrievalEvaluator:

    def evaluate(
        self,
        retrieved_chunks: list[str],
        reranked_chunks: list[str]
    ) -> RetrievalResult:

        retrieved = len(retrieved_chunks)
        reranked = len(reranked_chunks)

        if reranked > 0:
            avg_length = sum(len(c) for c in reranked_chunks) / reranked
        else:
            avg_length = 0

        score = min((reranked / 5) * 100, 100)

        return RetrievalResult(
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
            average_chunk_length=round(avg_length, 2),
            retrieval_score=round(score, 2),
        )