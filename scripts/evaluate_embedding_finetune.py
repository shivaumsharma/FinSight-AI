"""
evaluate_embedding_finetune.py

Fair before/after comparison of the fine-tuned embedding model
(scripts/finetune_embeddings.py's output) against the production
baseline (BAAI/bge-base-en-v1.5, untouched), scored against the SAME
held-out hand-labeled eval set used in scripts/evaluate_retrieval.py
(app/evaluation/retrieval_labels.py -- 7 queries, never used for
training).

Methodology: for each eval query, the baseline model's own top-20
retrieval (via the live production RAGPipeline) establishes a frozen
CANDIDATE POOL of real chunk texts -- this is the exact same pool the
hand labels were written against (chunk_id "TICKER_i" = the chunk at
baseline rank i). Both models then RE-RANK that same frozen pool by
their own cosine similarity to the query; only the ranking changes
between the two runs, not which chunks are in the pool. This isolates
the embedding model as the only variable, rather than letting a
different model retrieve a different set of candidates from the wider
corpus (which would leave chunks outside the original pool unlabeled
and make the two runs incomparable).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from app.rag.rag_pipeline import RAGPipeline
from app.reasoning.query_classifier import QueryClassifier
from app.reasoning.retrieval_planner import RetrievalPlanner
from app.evaluation.retrieval_labels import QUERIES, RELEVANT_BY_QUERY

from evaluate_retrieval import evaluate_ranking, fmt, K

FINETUNED_MODEL_PATH = str(Path(__file__).resolve().parent / "finetuned_bge_embeddings")
BASELINE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def rank_by_similarity(model, query, texts):
    query_emb = model.encode(query, normalize_embeddings=True)
    doc_embs = model.encode(texts, normalize_embeddings=True)
    scores = doc_embs @ query_emb
    order = np.argsort(-scores)
    return order


def main():
    pipeline = RAGPipeline()
    classifier = QueryClassifier()
    planner = RetrievalPlanner()

    print("Loading baseline and fine-tuned models...", file=sys.stderr)
    baseline_model = SentenceTransformer(BASELINE_MODEL_NAME)
    finetuned_model = SentenceTransformer(FINETUNED_MODEL_PATH)

    baseline_results = []
    finetuned_results = []

    print(f"{'Query':<20}{'BaselineP@5':>13}{'BaselineNDCG@5':>16}{'BaselineMRR':>13}   "
          f"{'FTunedP@5':>11}{'FTunedNDCG@5':>14}{'FTunedMRR':>11}")
    print("-" * 100)

    for q in QUERIES:
        ticker, question, qid = q["ticker"], q["question"], q["id"]
        relevant_ids = RELEVANT_BY_QUERY[qid]

        chunks_ingested, disclosure = pipeline.ingest_company_disclosure(ticker)
        if not disclosure:
            print(f"{qid:<20} [skip] no disclosure", file=sys.stderr)
            continue

        intent = classifier.classify(question)
        retrieval_query = planner.build(question, intent)

        # Baseline model's own retrieval defines the frozen candidate
        # pool (same as evaluate_retrieval.py) -- chunk_id "TICKER_i"
        # = the chunk at baseline rank i, matching how the labels in
        # retrieval_labels.py were created.
        raw_chunks = pipeline.query_pipeline(retrieval_query, ticker=ticker, n_results=20)
        pool_texts = [c["text"] for c in raw_chunks]
        pool_ids = [f"{ticker}_{i}" for i in range(len(pool_texts))]

        baseline_order = list(range(len(pool_texts)))  # already in baseline rank order
        baseline_ranked_ids = [pool_ids[i] for i in baseline_order]

        finetuned_order = rank_by_similarity(finetuned_model, question, pool_texts)
        finetuned_ranked_ids = [pool_ids[i] for i in finetuned_order]

        baseline_scores = evaluate_ranking(baseline_ranked_ids, relevant_ids)
        finetuned_scores = evaluate_ranking(finetuned_ranked_ids, relevant_ids)
        baseline_results.append(baseline_scores)
        finetuned_results.append(finetuned_scores)

        print(f"{qid:<20}{fmt(baseline_scores['precision_at_5']):>13}{fmt(baseline_scores['ndcg_at_5']):>16}"
              f"{fmt(baseline_scores['mrr']):>13}   "
              f"{fmt(finetuned_scores['precision_at_5']):>11}{fmt(finetuned_scores['ndcg_at_5']):>14}"
              f"{fmt(finetuned_scores['mrr']):>11}")

    def avg(results, key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    print("\n" + "=" * 70)
    print("AGGREGATE (mean across all queries, same held-out eval set)")
    print("=" * 70)
    print(f"{'Metric':<20}{'Baseline BGE':>16}{'Fine-tuned BGE':>18}")
    print("-" * 54)
    for label, key in [("Precision@5", "precision_at_5"), ("Recall@pool", "recall_at_pool"),
                        ("NDCG@5", "ndcg_at_5"), ("MRR", "mrr")]:
        print(f"{label:<20}{fmt(avg(baseline_results, key)):>16}{fmt(avg(finetuned_results, key)):>18}")


if __name__ == "__main__":
    main()
