"""
generate_embedding_training_data.py

Builds (query, relevant_chunk) training pairs for fine-tuning the
retrieval embedding model (Priority 2 -- "Option B" from the RAG
improvement discussion: real fine-tuning on labeled data, not
swapping in FinBERT's classification weights as embeddings directly,
which is a different, technically unsound idea ruled out earlier).

Methodology: synthetic query generation. For each real, already-
ingested SEC filing chunk, the local LLM (app/rag/report_generator.py)
is asked to write ONE natural question that chunk directly answers --
a standard, established technique in retrieval fine-tuning (see e.g.
InPars, Promptagator) for building training pairs without manual
labeling at scale. The CHUNK TEXT is always 100% real filing content;
only the QUESTION is LLM-synthesized. This is fundamentally different
from, and does not touch, app/evaluation/retrieval_labels.py's 7
hand-labeled eval queries -- those remain completely held out as the
only ground truth used to judge whether fine-tuning actually helped
(scripts/evaluate_embedding_finetune.py). Training tickers here are
deliberately disjoint from every eval-set ticker (AAPL, MSFT, NFLX,
COST, GOOGL) to avoid any contamination.

Output: scripts/embedding_training_pairs.jsonl
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.rag_pipeline import RAGPipeline
from app.rag.report_generator import ReportGenerator

# Disjoint from EVAL_TICKERS in app/evaluation/retrieval_labels.py
# (AAPL, MSFT, NFLX, COST, GOOGL) -- spans financials, energy,
# staples, healthcare, media, tech, industrials for topic diversity.
TRAIN_TICKERS = [
    "JPM", "BAC", "XOM", "CVX", "PG", "KO", "WMT", "HD",
    "PFE", "JNJ", "DIS", "ORCL", "CRM", "INTC", "CAT",
]

MIN_CHUNK_CHARS = 150  # shorter chunks (e.g. "Dimon added:") don't carry
                       # enough standalone content for a coherent question
MAX_CHUNKS_PER_TICKER = 20  # caps total generation time
OUTPUT_PATH = str(Path(__file__).resolve().parent / "embedding_training_pairs.jsonl")

PROMPT_TEMPLATE = """Below is an excerpt from a company's SEC filing or earnings release. Write ONE short, natural question that this excerpt directly and completely answers. The question should sound like something an investor would actually ask. Output ONLY the question, nothing else -- no preamble, no quotes, no notes.

Excerpt:
{text}

Question:"""


def clean_question(raw: str) -> str:
    """Takes the first line, cut at the first '?' if the model kept
    generating past the actual question (the short-chunk failure mode
    observed in testing: the model rambles into meta-commentary about
    its own instructions once the source excerpt runs out of real
    content to ask about)."""
    first_line = raw.strip().split("\n")[0].strip()
    if "?" in first_line:
        first_line = first_line[: first_line.index("?") + 1]
    return first_line.strip().strip('"')


def main():
    pipeline = RAGPipeline()
    gen = ReportGenerator()

    pairs = []
    for ticker in TRAIN_TICKERS:
        print(f"=== {ticker} ===", file=sys.stderr)
        try:
            pipeline.ingest_company_disclosure(ticker)
        except Exception as e:
            print(f"  [skip] ingestion failed: {e}", file=sys.stderr)
            continue

        stored = pipeline.vector_store.collection.get(where={"company": ticker})
        documents = stored.get("documents", [])
        long_enough = [d for d in documents if len(d) >= MIN_CHUNK_CHARS][:MAX_CHUNKS_PER_TICKER]
        print(f"  {len(documents)} chunks total, using {len(long_enough)}", file=sys.stderr)

        for i, text in enumerate(long_enough):
            prompt = PROMPT_TEMPLATE.format(text=text[:700])
            raw_question = gen.generate(prompt, max_new_tokens=40)
            question = clean_question(raw_question)
            if len(question) < 10 or "?" not in question:
                continue  # degenerate generation, skip rather than keep bad data
            pairs.append({"ticker": ticker, "query": question, "chunk_text": text})
            print(f"  [{i+1}/{len(long_enough)}] {question}", file=sys.stderr)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nTotal training pairs: {len(pairs)} -> {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
