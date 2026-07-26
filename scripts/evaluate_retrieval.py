"""
evaluate_retrieval.py

Real retrieval quality evaluation -- replaces app/evaluation/
retrieval_evaluator.py's existing "retrieval_score", which only checks
whether 5 reranked chunks came back at all (a coverage/count proxy,
not a relevance measurement -- see app/evaluation/retrieval_evaluator.py's
own `score = min((reranked / 5) * 100, 100)`).

Runs the real production query path (QueryClassifier -> RetrievalPlanner
-> RAGPipeline.query_pipeline -> EvidenceReranker) against 7 real
(ticker, question) pairs spanning 5 companies, and scores both the raw
retrieval (top-20) and the reranked output (top-5) against hand-labeled
relevance judgments (app/evaluation/retrieval_labels.py -- every
judgment was read and labeled by hand against the query's actual
information need, not estimated).

Metrics: Precision@5, Recall@20 (recall against the pool of chunks
that were actually retrieved and judged -- NOT recall against every
possibly-relevant sentence in the source document, which would need
exhaustive judging of content that was never retrieved at all; this is
the standard "pooled judgment" limitation of small-scale IR eval,
stated rather than glossed over), NDCG@5, and MRR (mean reciprocal
rank of the first relevant result).

Reports raw retrieval and reranked retrieval SEPARATELY, on purpose:
the reranker (app/rag/reranker.py) has been running in production this
whole time with no measurement of whether it actually improves
relevance over raw embedding retrieval. This eval is the first time
that's been checked.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.rag_pipeline import RAGPipeline
from app.reasoning.query_classifier import QueryClassifier
from app.reasoning.retrieval_planner import RetrievalPlanner
from app.rag.reranker import EvidenceReranker
from app.evaluation.retrieval_labels import QUERIES, RELEVANT_BY_QUERY

K = 5  # matches EvidenceReranker's top_k=5, the actual production cutoff


def dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(ranked_relevances, k):
    actual = ranked_relevances[:k]
    ideal = sorted(ranked_relevances, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return None  # no relevant items exist in this pool at all -- NDCG undefined, not zero
    return dcg(actual) / ideal_dcg


def reciprocal_rank(ranked_relevances):
    for idx, rel in enumerate(ranked_relevances):
        if rel:
            return 1.0 / (idx + 1)
    return 0.0


def evaluate_ranking(chunk_ids_in_rank_order, relevant_ids):
    """Given a ranked list of chunk_ids and the set judged relevant,
    returns precision@5, recall@len(list) (pooled), ndcg@5, and MRR."""
    relevances = [1 if cid in relevant_ids else 0 for cid in chunk_ids_in_rank_order]

    top_k = relevances[:K]
    precision_at_k = sum(top_k) / K if len(relevances) >= K else (sum(top_k) / len(top_k) if top_k else None)

    total_relevant_in_pool = len(relevant_ids)
    recall_at_pool = (sum(relevances) / total_relevant_in_pool) if total_relevant_in_pool > 0 else None

    ndcg = ndcg_at_k(relevances, K)
    mrr = reciprocal_rank(relevances)

    return {
        "precision_at_5": precision_at_k,
        "recall_at_pool": recall_at_pool,
        "ndcg_at_5": ndcg,
        "mrr": mrr,
        "n_relevant_in_pool": total_relevant_in_pool,
        "n_retrieved": len(relevances),
    }


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def main():
    pipeline = RAGPipeline()
    classifier = QueryClassifier()
    planner = RetrievalPlanner()
    reranker = EvidenceReranker()

    raw_results = []
    reranked_results = []

    print(f"{'Query':<20}{'RawP@5':>9}{'RawNDCG@5':>11}{'RawMRR':>9}   "
          f"{'RerankP@5':>10}{'RerankNDCG@5':>14}{'RerankMRR':>11}")
    print("-" * 84)

    for q in QUERIES:
        ticker, question, qid = q["ticker"], q["question"], q["id"]
        relevant_ids = RELEVANT_BY_QUERY[qid]

        chunks_ingested, disclosure = pipeline.ingest_company_disclosure(ticker)
        if not disclosure:
            print(f"{qid:<20} [skip] no disclosure available", file=sys.stderr)
            continue

        intent = classifier.classify(question)
        retrieval_query = planner.build(question, intent)
        raw_chunks = pipeline.query_pipeline(retrieval_query, ticker=ticker, n_results=20)
        reranked_chunks = reranker.rerank(question, raw_chunks, top_k=K)

        # Same positional chunk_id scheme used when the labels were
        # created: rank i (0-indexed) in THIS query's own raw result
        # order -> f"{ticker}_{i}". Retrieval is deterministic (same
        # stored embeddings, same query text), so this reproduces the
        # exact same mapping labeling was done against.
        raw_ids = [f"{ticker}_{i}" for i in range(len(raw_chunks))]
        raw_text_to_id = {c["text"]: raw_ids[i] for i, c in enumerate(raw_chunks)}
        reranked_ids = [raw_text_to_id[c["text"]] for c in reranked_chunks]

        raw_scores = evaluate_ranking(raw_ids, relevant_ids)
        reranked_scores = evaluate_ranking(reranked_ids, relevant_ids)
        raw_results.append(raw_scores)
        reranked_results.append(reranked_scores)

        print(f"{qid:<20}{fmt(raw_scores['precision_at_5']):>9}{fmt(raw_scores['ndcg_at_5']):>11}"
              f"{fmt(raw_scores['mrr']):>9}   "
              f"{fmt(reranked_scores['precision_at_5']):>10}{fmt(reranked_scores['ndcg_at_5']):>14}"
              f"{fmt(reranked_scores['mrr']):>11}")

    def avg(results, key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n" + "=" * 84)
    print("AGGREGATE (mean across all queries)")
    print("=" * 84)
    print(f"{'Metric':<20}{'Raw retrieval':>18}{'Reranked (production)':>26}")
    print("-" * 64)
    for label, key in [("Precision@5", "precision_at_5"), ("Recall@pool", "recall_at_pool"),
                        ("NDCG@5", "ndcg_at_5"), ("MRR", "mrr")]:
        print(f"{label:<20}{fmt(avg(raw_results, key)):>18}{fmt(avg(reranked_results, key)):>26}")

    print(f"\nQueries evaluated: {len(raw_results)}")
    zero_relevant = [q['id'] for q in QUERIES if len(RELEVANT_BY_QUERY[q['id']]) == 0]
    if zero_relevant:
        print(f"Queries with NO relevant content in the ingested document at all: {zero_relevant} "
              f"(excluded from recall/NDCG/MRR averages above -- undefined, not zero, when nothing "
              f"relevant exists to rank)")


if __name__ == "__main__":
    main()
