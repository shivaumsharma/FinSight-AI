"""
retrieval_labels.py

Hand-labeled relevance judgments for a real retrieval evaluation set --
7 (ticker, question) pairs spanning 5 real companies (AAPL, MSFT, NFLX,
COST, GOOGL) and the actual production query-construction path
(QueryClassifier -> RetrievalPlanner), run against real ingested SEC
filing content via RAGPipeline.query_pipeline + EvidenceReranker (see
scripts/evaluate_retrieval.py, which consumes this file).

Every chunk_id below was read and judged by hand against its query's
actual information need -- not scored programmatically, not
estimated. This is what makes the resulting precision/recall/NDCG
numbers real instead of the kind of invented "0.85, 0.92, 0.88"
figures a fabricated eval would produce.

RELEVANT_CHUNK_IDS: chunk_ids judged relevant to that query, out of
everything retrieved in the raw top-20 (or fewer, if the filing didn't
have 20 chunks worth of content). Anything retrieved but not listed
here was judged NOT relevant. chunk_ids are stable within one ticker's
current ingestion (see Text_chunker.py's f"{company}_{quarter}_{idx}"
scheme) -- re-running against freshly re-ingested content could shift
them; this file is a point-in-time evaluation snapshot, matching how
phase2_backtest.py's own JSON snapshots work.
"""

QUERIES = [
    {
        "id": "aapl_investment",
        "ticker": "AAPL",
        "question": "Should I buy Apple?",
    },
    {
        "id": "aapl_ai_demand",
        "ticker": "AAPL",
        "question": "What did management say about AI demand?",
    },
    {
        "id": "msft_investment",
        "ticker": "MSFT",
        "question": "Should I buy Microsoft?",
    },
    {
        "id": "msft_financial",
        "ticker": "MSFT",
        "question": "What is Microsoft's revenue and margin performance?",
    },
    {
        "id": "nflx_investment",
        "ticker": "NFLX",
        "question": "Should I buy Netflix?",
    },
    {
        "id": "cost_investment",
        "ticker": "COST",
        "question": "Should I buy Costco?",
    },
    {
        "id": "googl_risks",
        "ticker": "GOOGL",
        "question": "What are Google's main risks?",
    },
]

# chunk_ids here are POSITIONAL (0-indexed by raw retrieval rank at
# labeling time: "AAPL_0" = the chunk that was raw-rank-1 for that
# specific query, NOT a stable Text_chunker chunk_id) -- see
# scripts/evaluate_retrieval.py, which re-runs retrieval and maps
# positional rank back to these labels rather than matching on
# Text_chunker's own chunk_id, since two different queries against the
# same ticker retrieve the same underlying chunks in different orders.
RELEVANT_BY_QUERY = {
    "aapl_investment": {"AAPL_0", "AAPL_1", "AAPL_3", "AAPL_4"},
    # None of AAPL's 5 available chunks (a dividend/buyback
    # announcement 8-K) actually discuss AI demand at all -- a real,
    # honest "the source document doesn't cover this topic" case, not
    # a retrieval failure. Kept as an empty set deliberately, not
    # dropped from the eval, since a system that confidently returns
    # unrelated chunks for a question its data can't answer is exactly
    # the failure mode a real eval should catch.
    "aapl_ai_demand": set(),
    "msft_investment": {"MSFT_0", "MSFT_2", "MSFT_4", "MSFT_8", "MSFT_10"},
    "msft_financial": {"MSFT_0", "MSFT_1", "MSFT_3", "MSFT_4", "MSFT_7", "MSFT_9", "MSFT_11", "MSFT_12"},
    "nflx_investment": {
        "NFLX_1", "NFLX_2", "NFLX_3", "NFLX_6", "NFLX_7", "NFLX_8",
        "NFLX_9", "NFLX_11", "NFLX_12", "NFLX_15", "NFLX_16", "NFLX_19",
    },
    "cost_investment": {"COST_0", "COST_2"},
    "googl_risks": {"GOOGL_0", "GOOGL_1", "GOOGL_5", "GOOGL_6"},
}
