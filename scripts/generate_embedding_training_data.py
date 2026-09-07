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

WHY THE FIRST 194-PAIR RUN MADE RETRIEVAL WORSE (EVALUATION.md
section 3), root-caused rather than guessed at: two compounding
problems, both fixed here.
  1. Too little data: 194 pairs, batch_size=16 -> ~12 batches/epoch,
     a small and easy-to-overfit in-batch-negative pool for
     MultipleNegativesRankingLoss (few negatives per batch means the
     model doesn't have to learn fine-grained distinctions to get
     every batch "right"). Fixed by generating from more tickers and
     more chunks per ticker (below).
  2. Leaked answers, confirmed by directly inspecting the actual
     pairs, not assumed: 31% of the original 194 questions contained
     a number (a $ figure, a %, a specific dollar amount) that ALSO
     appeared verbatim in the chunk -- e.g. "What is the reported
     overhead ratio of 48%..." asked about a chunk whose answer IS
     48%. A question that hands over its own answer trains the model
     to match on literal number/keyword overlap, not semantic
     understanding -- exactly the "memorized surface patterns instead
     of learning financial concepts" failure mode, and a large enough
     fraction of the data (31%) to plausibly explain the collapse on
     its own. Fixed two ways: a stronger prompt instruction (below),
     AND a post-generation filter that discards any pair where the
     leak still slipped through (see _leaks_answer) -- not trusting
     the prompt alone to fully prevent what's a well-documented LLM
     tendency to follow the letter of "don't copy numbers" while still
     restating them in a lightly reworded form.

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
# staples, healthcare, media, tech, industrials, retail, autos,
# aerospace, pharma for topic diversity. Expanded from the original
# 15 tickers (194 pairs -> too few, see module docstring) -- 10 more
# added, still disjoint from the eval set.
TRAIN_TICKERS = [
    "JPM", "BAC", "XOM", "CVX", "PG", "KO", "WMT", "HD",
    "PFE", "JNJ", "DIS", "ORCL", "CRM", "INTC", "CAT",
    "NKE", "SBUX", "LOW", "TGT", "UNH", "MRK", "BA", "GE", "F", "GM",
]

MIN_CHUNK_CHARS = 150  # shorter chunks (e.g. "Dimon added:") don't carry
                       # enough standalone content for a coherent question
MAX_CHUNKS_PER_TICKER = 35  # was 20 -- raised alongside more tickers to
                            # grow total volume (see module docstring's
                            # "too little data" root cause)
OUTPUT_PATH = str(Path(__file__).resolve().parent / "embedding_training_pairs.jsonl")

# Explicit, repeated instruction against copying numbers -- softer
# phrasing ("avoid quoting exact figures") was tried conceptually and
# rejected in favor of this more forceful version specifically because
# the failure mode observed in the original data wasn't subtle (a
# question that literally contains "48%" right next to "overhead
# ratio"), so the instruction needs to be equally unambiguous, not a
# polite suggestion an LLM can partially ignore.
PROMPT_TEMPLATE = """Below is an excerpt from a company's SEC filing or earnings release. Write ONE short, natural question that this excerpt directly and completely answers. The question should sound like something an investor would actually ask BEFORE knowing the answer.

CRITICAL RULE: Do NOT include any specific number, percentage, or dollar figure from the excerpt in your question. Ask ABOUT the fact conceptually (e.g. "What was JPMorgan's efficiency ratio this quarter?") -- NEVER restate the number itself (e.g. NOT "What is the 48% overhead ratio?"). If you cannot write a question about this excerpt without quoting a number from it, write a more general question about the topic instead.

Output ONLY the question, nothing else -- no preamble, no quotes, no notes.

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


# Matches a percentage, a decimal, a comma-grouped number, or a bare
# multi-digit integer -- deliberately permissive (better to discard a
# borderline-fine pair than keep one that leaks) since the whole point
# is a QC backstop, not a precision-tuned classifier. A 1-digit number
# ("3 segments") is excluded -- too likely to be a false-positive
# coincidence (page numbers, footnote markers) rather than a genuine
# leaked fact.
_NUMBER_PATTERN = re.compile(r"\d[\d,.]*%?")


def _leaks_answer(question: str, chunk_text: str) -> bool:
    """True if a 2+-digit number in the question also appears verbatim
    in the chunk -- see module docstring's root-cause #2. Confirmed
    against the original data before trusting this as the filter:
    31% of the first 194 pairs (60 of them) leaked this way."""
    q_nums = set(_NUMBER_PATTERN.findall(question))
    c_nums = set(_NUMBER_PATTERN.findall(chunk_text))
    leaked = {n for n in q_nums if n in c_nums and len(re.sub(r"[.,%]", "", n)) >= 2}
    return bool(leaked)


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

        leaked_count = 0
        for i, text in enumerate(long_enough):
            prompt = PROMPT_TEMPLATE.format(text=text[:700])
            raw_question = gen.generate(prompt, max_new_tokens=40)
            question = clean_question(raw_question)
            if len(question) < 10 or "?" not in question:
                continue  # degenerate generation, skip rather than keep bad data
            if _leaks_answer(question, text):
                # QC backstop -- see _leaks_answer's own docstring for
                # why the prompt instruction alone isn't trusted to
                # fully prevent this. Discarded, not "fixed" by
                # stripping the number out: a question rewritten after
                # the fact to remove a number it was built around often
                # reads as awkward or ungrammatical, which is its own
                # form of bad training data.
                leaked_count += 1
                print(f"  [{i+1}/{len(long_enough)}] [discarded -- leaks answer] {question}", file=sys.stderr)
                continue
            pairs.append({"ticker": ticker, "query": question, "chunk_text": text})
            print(f"  [{i+1}/{len(long_enough)}] {question}", file=sys.stderr)
        if leaked_count:
            print(f"  {leaked_count}/{len(long_enough)} discarded for leaking the answer", file=sys.stderr)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nTotal training pairs: {len(pairs)} -> {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
