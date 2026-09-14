"""
generate_embedding_training_data.py

Builds (query, relevant_chunk) training pairs for fine-tuning the
retrieval embedding model (Priority 2 -- "Option B" from the RAG
improvement discussion: real fine-tuning on labeled data, not
swapping in FinBERT's classification weights as embeddings directly,
which is a different, technically unsound idea ruled out earlier).

Methodology: synthetic query generation. For each real, already-
ingested SEC filing chunk, an LLM is asked to write ONE natural
question that chunk directly answers -- a standard, established
technique in retrieval fine-tuning (see e.g. InPars, Promptagator) for
building training pairs without manual labeling at scale. The CHUNK
TEXT is always 100% real filing content; only the QUESTION is
LLM-synthesized. This is fundamentally different from, and does not
touch, app/evaluation/retrieval_labels.py's 7 hand-labeled eval
queries -- those remain completely held out as the only ground truth
used to judge whether fine-tuning actually helped
(scripts/evaluate_embedding_finetune.py). Training tickers here are
deliberately disjoint from every eval-set ticker (AAPL, MSFT, NFLX,
COST, GOOGL) to avoid any contamination.

THIRD ROOT CAUSE, found while scaling past 328 pairs (see below) and
worse than either of the first two: this script used to import
app/rag/report_generator.py's ReportGenerator directly -- the local
Qwen2.5-1.5B-Instruct model, hardcoded, regardless of this project's
own .env LLM_PROVIDER setting (default "hosted", meaning production
itself runs a much larger hosted model -- see app/core/llm_provider.py).
Generating the sector-stratified 125-ticker set with the old code
surfaced a serious, silent failure the 25-ticker run never happened to
show clearly: the small model would confidently hallucinate "JPMorgan"
as the subject of a question about a COMPLETELY UNRELATED company
(e.g. an ExxonMobil carbon-capture excerpt -> "...explored by
JPMorgan?"). Confirmed directly, not assumed: re-ran the exact failing
XOM chunk (which never mentions JPMorgan anywhere in its text) through
a fresh, single, stateless call to the local model -- reproduced the
identical hallucination -- then through get_llm_provider()'s real
hosted model on the identical prompt, which wrote a correct,
appropriately-scoped question with no fabricated company. This isn't a
context-caching bug (a fresh single call still failed); it's a small
model defaulting to the most textually common finance entity in its
training data whenever a chunk's own visible text doesn't explicitly
restate the company name (boilerplate MD&A/forward-looking-statement
sections often don't). A wrong-company question is worse than a
leaked-number one: it doesn't just hand the model a shortcut, it's an
outright false (query, chunk) pair that teaches retrieval AWAY from
correct matches. Fixed by routing through get_llm_provider() (the same
abstraction production uses) instead of importing ReportGenerator
directly, so this script's data quality tracks whatever model this
project is actually configured to run -- plus a QC backstop
(_mentions_wrong_company) as defense-in-depth, matching this module's
existing "don't trust the prompt/model alone" posture toward
_leaks_answer.

WHY THE FIRST 194-PAIR RUN MADE RETRIEVAL WORSE (EVALUATION.md
section 3), root-caused rather than guessed at: two compounding
problems, both fixed here.
  1. Too little data: 194 pairs, batch_size=16 -> ~12 batches/epoch,
     a small and easy-to-overfit in-batch-negative pool for
     MultipleNegativesRankingLoss (few negatives per batch means the
     model doesn't have to learn fine-grained distinctions to get
     every batch "right"). First fix (194 -> 328 pairs, 15 -> 25
     tickers) was directionally right but still too small -- see
     "SCALING PAST 328 PAIRS" below for the second, larger pass.
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

SCALING PAST 328 PAIRS: fixing both causes above closed part of the
gap to baseline but the fine-tune still lost on every metric
(EVALUATION.md section 3) -- 328 pairs across 25 tickers is still tiny
by normal embedding-fine-tuning standards, AND a third, separate issue
was found in finetune_embeddings.py itself (that script trained
blindly for 4 fixed epochs with no held-out signal at all; see its own
docstring for the loss-collapse evidence). This pass fixes the data
side of that: TRAIN_TICKERS is no longer a short hand-picked list --
it's a large, sector-stratified sample drawn from
scripts/ticker_universe.json, the same 1,002-ticker broad universe
already built for this project's backtests, reused here instead of
hand-curating a second ticker list. Stratifying by GICS sector (not a
flat random sample) matters for a contrastive loss specifically:
MultipleNegativesRankingLoss's only source of negatives is whatever
else lands in the same training batch, so a training set clustered in
2-3 sectors would mostly hand the model easy, same-sector-look-alike
negatives across a run; spreading tickers evenly across every sector
gives it a genuinely varied negative pool to learn fine-grained
distinctions from, batch to batch.

Output: scripts/embedding_training_pairs.jsonl
"""

import concurrent.futures
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.core.llm_provider import get_llm_provider
from app.rag.rag_pipeline import RAGPipeline

# Must stay disjoint from the production hand-labeled eval set
# (app/evaluation/retrieval_labels.py) -- those 5 tickers are the only
# ground truth used to judge whether fine-tuning helped, so none of
# them can also appear in training.
EVAL_TICKERS = {"AAPL", "MSFT", "NFLX", "COST", "GOOGL"}

# The original, hand-picked 25-ticker list from the first scale-up
# (194 -> 328 pairs). Kept as-is rather than dropped -- these were
# already confirmed to ingest cleanly, and there's no reason to throw
# away working data sources when growing the set further.
_ORIGINAL_TICKERS = [
    "JPM", "BAC", "XOM", "CVX", "PG", "KO", "WMT", "HD",
    "PFE", "JNJ", "DIS", "ORCL", "CRM", "INTC", "CAT",
    "NKE", "SBUX", "LOW", "TGT", "UNH", "MRK", "BA", "GE", "F", "GM",
]

TICKER_UNIVERSE_PATH = str(Path(__file__).resolve().parent / "ticker_universe.json")
TARGET_NEW_TICKERS = 100  # on top of the 25 original -> 125 total
SECTOR_SAMPLE_SEED = 42


def _build_train_tickers() -> list:
    """Sector-stratified sample of TARGET_NEW_TICKERS additional
    tickers from the broad backtest universe, disjoint from both the
    eval set and the original 25. Round-robins across sectors (each
    internally shuffled with a fixed seed for reproducibility) so the
    result spans every GICS sector roughly evenly rather than
    whichever sectors happen to sort first alphabetically."""
    with open(TICKER_UNIVERSE_PATH, encoding="utf-8") as f:
        universe = json.load(f)

    excluded = EVAL_TICKERS | set(_ORIGINAL_TICKERS)
    by_sector = {}
    for ticker, sector_label in universe.items():
        if ticker in excluded:
            continue
        sector = sector_label.split(" (")[0]
        by_sector.setdefault(sector, []).append(ticker)

    rng = random.Random(SECTOR_SAMPLE_SEED)
    for tickers in by_sector.values():
        rng.shuffle(tickers)

    sectors = sorted(by_sector.keys())
    new_tickers = []
    idx = 0
    while len(new_tickers) < TARGET_NEW_TICKERS and any(by_sector[s] for s in sectors):
        sector = sectors[idx % len(sectors)]
        if by_sector[sector]:
            new_tickers.append(by_sector[sector].pop())
        idx += 1

    return _ORIGINAL_TICKERS + new_tickers


TRAIN_TICKERS = _build_train_tickers()

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
#
# The illustrative example used to name a real company ("What was
# JPMorgan's efficiency ratio this quarter?") -- found, while
# diagnosing the third root cause above, to be a likely direct cause
# of it: "efficiency ratio" (this example's own made-up metric)
# showed up verbatim, still attributed to "JPMorgan," in dozens of
# hallucinated questions about companies that have nothing to do with
# banking. A real name inside a "good example" is a classic prompt-
# anchoring trap -- the model latches onto the example's specific
# entity as a template to imitate, especially on a source excerpt
# (boilerplate MD&A/forward-looking-statements text) that never
# restates the actual company's name itself. Fixed by making the
# example generic ("the company's") instead of naming any real one --
# this is a model-agnostic fix (unlike the provider switch above,
# which only fixes it for whichever model runs this prompt) and should
# hold regardless of which LLM ends up generating these questions.
PROMPT_TEMPLATE = """Below is an excerpt from a company's SEC filing or earnings release. Write ONE short, natural question that this excerpt directly and completely answers. The question should sound like something an investor would actually ask BEFORE knowing the answer.

CRITICAL RULE: Do NOT include any specific number, percentage, or dollar figure from the excerpt in your question. Ask ABOUT the fact conceptually (e.g. "What was the company's efficiency ratio this quarter?") -- NEVER restate the number itself (e.g. NOT "What is the 48% overhead ratio?"). If this excerpt does not explicitly name the company, refer to it generically as "the company" or "this company" -- do NOT guess or substitute a different, unrelated company's name. If you cannot write a question about this excerpt without quoting a number from it, write a more general question about the topic instead.

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


# Deliberately narrow, not a general-purpose NER cross-check: this is a
# QC backstop targeted at the SPECIFIC hallucination pattern confirmed
# in this module's docstring (root cause #3) -- a handful of
# extremely textually-common finance entities a weak generation showed
# a real tendency to default to when a chunk's own text doesn't name
# its company. A question naming any of these, on a ticker that isn't
# actually one of them, is treated as a misattributed pair and
# discarded the same way a leaked number is.
_HALLUCINATION_ATTRACTORS = {
    "JPM": ("JPMorgan", "JP Morgan", "J.P. Morgan"),
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "GOOGL": ("Google", "Alphabet"),
    "AMZN": ("Amazon",),
}


def _mentions_wrong_company(question: str, ticker: str) -> bool:
    """True if the question names one of _HALLUCINATION_ATTRACTORS'
    companies while `ticker` isn't that company -- see root cause #3."""
    for attractor_ticker, names in _HALLUCINATION_ATTRACTORS.items():
        if ticker == attractor_ticker:
            continue
        if any(name in question for name in names):
            return True
    return False


GENERATION_MAX_NEW_TOKENS = 130  # was 40 against the old local model;
    # get_llm_provider()'s hosted gpt-oss-120b spends real tokens on
    # hidden reasoning before its actual answer (confirmed live: 34
    # reasoning tokens on a representative prompt, but a smoke test at
    # 100 still occasionally truncated mid-question on a harder
    # prompt) -- 130 leaves real margin rather than cutting it close.
    # See root cause #3 above for why this switched providers at all.

# FOURTH ROOT CAUSE, found the hard way: a full 125-ticker run was left
# running overnight and was found ~11 hours later stuck at ticker 51,
# with the underlying process still alive but producing nothing. The
# machine's live TCP state showed one connection stuck ESTABLISHED
# with no data flowing and a second stuck in CLOSE_WAIT -- consistent
# with a laptop sleep/resume cycle leaving a socket in a state the OS
# still reports as open but that will never receive another byte.
# HostedProvider's own `timeout=60` (app/core/llm_provider.py) doesn't
# reliably catch this class of hang: `requests`' read-timeout only
# fires on a gap between bytes, and a genuinely dead-but-not-yet-RST
# socket can sit past that indefinitely (Windows' own default TCP
# keepalive is 2 HOURS). Two independent fixes, not one:
#   1. A hard, script-level watchdog (_generate_with_watchdog below) --
#      independent of whatever the shared provider's own timeout does
#      or doesn't catch, so a single hung call can cost at most
#      GENERATE_WATCHDOG_SECONDS, never the rest of the night.
#   2. Incremental output writes (flushed after every ticker, not once
#      at the very end) -- so a hang, crash, or a laptop actually
#      going to sleep despite being asked not to still leaves every
#      completed ticker's pairs safely on disk instead of losing all
#      of it, which is exactly what happened to the first attempt at
#      this scale.
GENERATE_WATCHDOG_SECONDS = 120
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _generate_with_watchdog(gen, prompt, max_new_tokens):
    """None on timeout (treated the same as a degenerate generation by
    the caller) instead of raising -- one stuck chunk should cost at
    most GENERATE_WATCHDOG_SECONDS and move on, not take the whole run
    down. The submitted call keeps running in its own thread even
    after the timeout (there's no way to forcibly kill it), which is a
    real but acceptable leak for a script that runs once and exits --
    not something this executor's single worker slot can ever recover
    from mid-run, which is exactly why every subsequent call is
    submitted to a NEW single-use executor rather than reused (see
    call site)."""
    future = _executor.submit(gen.generate, prompt, max_new_tokens=max_new_tokens)
    try:
        return future.result(timeout=GENERATE_WATCHDOG_SECONDS)
    except concurrent.futures.TimeoutError:
        return None


def main():
    global _executor
    pipeline = RAGPipeline()
    gen = get_llm_provider()

    total_pairs = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
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
            wrong_company_count = 0
            timed_out_count = 0
            ticker_pairs = []
            for i, text in enumerate(long_enough):
                prompt = PROMPT_TEMPLATE.format(text=text[:700])
                raw_question = _generate_with_watchdog(gen, prompt, GENERATION_MAX_NEW_TOKENS)
                if raw_question is None:
                    # The stuck worker thread is abandoned -- a fresh
                    # single-use executor for every call (not one
                    # reused across the whole run) means a hang here
                    # can never block the NEXT call from being
                    # submitted, since it isn't waiting on the same
                    # worker slot.
                    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    timed_out_count += 1
                    print(f"  [{i+1}/{len(long_enough)}] [discarded -- generation timed out after {GENERATE_WATCHDOG_SECONDS}s]", file=sys.stderr)
                    continue
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
                if _mentions_wrong_company(question, ticker):
                    # Second QC backstop -- see root cause #3 above.
                    wrong_company_count += 1
                    print(f"  [{i+1}/{len(long_enough)}] [discarded -- wrong company] {question}", file=sys.stderr)
                    continue
                ticker_pairs.append({"ticker": ticker, "query": question, "chunk_text": text})
                print(f"  [{i+1}/{len(long_enough)}] {question}", file=sys.stderr)
            if leaked_count:
                print(f"  {leaked_count}/{len(long_enough)} discarded for leaking the answer", file=sys.stderr)
            if wrong_company_count:
                print(f"  {wrong_company_count}/{len(long_enough)} discarded for naming the wrong company", file=sys.stderr)
            if timed_out_count:
                print(f"  {timed_out_count}/{len(long_enough)} discarded for a hung generation call", file=sys.stderr)

            for pair in ticker_pairs:
                out_f.write(json.dumps(pair) + "\n")
            out_f.flush()
            total_pairs += len(ticker_pairs)

    print(f"\nTotal training pairs: {total_pairs} -> {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
