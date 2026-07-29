"""
compare_local_vs_hosted_narrative.py

Required manual verification before LLM_PROVIDER=hosted is used for
anything real -- not optional polish. See app/core/llm_provider.py's
module docstring for the risk this exists to check: every prompt in
this codebase is a raw text-continuation prompt aimed at llama.cpp's
completion API, never chat-formatted. A hosted OpenAI-compatible CHAT
completions endpoint necessarily receives that same text as a single
`user` message and may respond AS AN ASSISTANT ANSWERING it, not
continue it the way the local completion API does -- different
framing, unwanted preamble, or an outright refusal to "just continue."
Stubbed unit tests (app/tests/test_llm_provider.py) verify the
provider interface itself works; they cannot verify this, because it's
a question about output quality, not code correctness.

Runs the SAME real narrative prompt -- built from a real, live report
via the actual pipeline, not a toy prompt, so this is representative
of what production actually sends -- through LocalLlamaProvider and
HostedProvider, and writes the prompt plus both full outputs to files
for a human to read side by side. There is no automated pass/fail
here: "is this narrative acceptably close to the local one" isn't
something this script can judge. Reading it is the verification.

Usage:
    python scripts/compare_local_vs_hosted_narrative.py [TICKER]

    Runs the local half unconditionally. Runs the hosted half only if
    LLM_BASE_URL/LLM_API_KEY/LLM_MODEL are set (a real hosted account
    is an external cost/decision this script won't default into place
    -- if they're unset, it prints how to set them and stops there).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm_provider import HostedProvider, LLMProviderError, LocalLlamaProvider
from app.core.research_context import ResearchContext
from app.reporting.narrative_builder import _build_prompt
from app.reporting.report_data_builder import build_report_data
from app.tools.institutional_consensus_tool import InstitutionalConsensusTool
from app.tools.market_data_tool import MarketDataTool
from app.tools.news_tool import NewsTool
from app.tools.rag_tool import RAGTool
from app.tools.sentiment_tool import SentimentTool
from app.tools.valuation_tool import ValuationTool

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "narrative_comparison"


def build_real_prompt(ticker: str) -> str:
    """
    Runs the real evidence-gathering tools (market data, valuation,
    RAG, sentiment, news, institutional consensus) against a real
    ResearchContext, then builds the exact same prompt
    narrative_builder.build_narrative_sections() would send in
    production -- reusing _build_prompt directly, not a hand-written
    approximation of it, so this comparison can't silently drift from
    what the real prompt actually looks like.
    """
    context = ResearchContext(ticker=ticker, question=f"Should I invest in {ticker}?")
    for tool in (
        MarketDataTool(), ValuationTool(), RAGTool(), SentimentTool(),
        InstitutionalConsensusTool(), NewsTool(),
    ):
        tool.run(context)

    report_data = build_report_data(context)
    return _build_prompt(context, report_data)


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

    print(f"Building a real narrative prompt for {ticker} (runs the real pipeline -- market data, "
          f"valuation, RAG, sentiment, news, consensus -- may take a minute)...")
    prompt = build_real_prompt(ticker)
    print(f"Prompt built: {len(prompt)} chars.\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"{ticker}_prompt.txt").write_text(prompt, encoding="utf-8")

    print("Running LOCAL provider (llama.cpp)...")
    local_start = time.time()
    local_output = LocalLlamaProvider().generate(prompt, max_new_tokens=700)
    local_elapsed = time.time() - local_start
    local_path = OUTPUT_DIR / f"{ticker}_local_narrative.txt"
    local_path.write_text(local_output, encoding="utf-8")
    print(f"  -> {local_path}  ({local_elapsed:.1f}s)")

    print("\nRunning HOSTED provider...")
    try:
        hosted_provider = HostedProvider()
    except LLMProviderError as e:
        print(f"  SKIPPED -- {e}")
        print("  Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL (see .env.example) to run this half.")
        print(f"\nOnly the local narrative was generated: {local_path}")
        return

    hosted_start = time.time()
    try:
        hosted_output = hosted_provider.generate(prompt, max_new_tokens=700)
    except LLMProviderError as e:
        print(f"  FAILED -- {e}")
        return
    hosted_elapsed = time.time() - hosted_start

    hosted_path = OUTPUT_DIR / f"{ticker}_hosted_narrative.txt"
    hosted_path.write_text(hosted_output, encoding="utf-8")
    print(f"  -> {hosted_path}  ({hosted_elapsed:.1f}s)")
    usage = hosted_provider.get_last_usage()
    if usage:
        print(f"  token usage: {usage}")

    print(f"\nWall-clock: local {local_elapsed:.1f}s vs hosted {hosted_elapsed:.1f}s")

    print(
        f"\n{'=' * 70}\n"
        f"Read these side by side before LLM_PROVIDER=hosted is used for anything real:\n"
        f"  prompt sent:    {OUTPUT_DIR / f'{ticker}_prompt.txt'}\n"
        f"  local output:   {local_path}\n"
        f"  hosted output:  {hosted_path}\n"
        f"{'=' * 70}\n"
        f"Specifically look for: does the hosted output continue the report template the way the "
        f"local one does, or does it respond conversationally / add preamble / refuse to \"just "
        f"continue\"? If the shape is meaningfully different, the prompts need chat-formatting -- "
        f"a follow-up pass, not something to discover after switching providers in production."
    )


if __name__ == "__main__":
    main()
