"""
benchmark_orchestration.py

Runs the same real questions through both orchestrators -- the
hand-rolled ResearchAgent (app/agents/research_agent.py) and its
LangGraph port (app/agents/langgraph_agent.py) -- and reports whether
they (a) visit the exact same tools in the exact same order and
(b) how their wall-clock latency compares. This is the evidence behind
"benchmarked framework orchestration against a hand-rolled controller,"
not an assumption.

Three modes
-----------
--pure (default): EVERY tool (all 9, not just report/evaluation) is a
no-op stand-in -- zero network calls, zero model inference, on both
orchestrators. This is the actual "framework overhead" question in
isolation: pure Python object dispatch through a StateGraph's Pregel
executor vs. a plain `for` loop, with the real tools' own execution
time (which neither orchestrator has any control over) removed from
the measurement entirely. Run N times per question (default 20) and
reports the median, since even pure-Python timing has some jitter.

--lite: report_tool/evaluation_tool are no-op stand-ins, but the other
6 tools run for real (real yfinance/SEC EDGAR/Finnhub/FinBERT calls).
Useful for seeing real tool_trace correctness against live data, but
NOT a clean orchestration-overhead measurement -- see the empirical
note below.

--full: no stand-ins anywhere, true end-to-end latency including the
~65s LLM narrative call. Slow (multiple minutes per query).

Why --pure exists (found empirically, not assumed up front): an
initial --lite run on real AAPL/MSFT queries produced a 964s outlier
for one ResearchAgent run against a 32s LangGraph run on the same
ticker -- not a real 30x orchestration difference, but a stalled SEC/
yfinance call on that particular run (a benign "possibly delisted, no
price data found" transient warning showed up in the same run). Real
network I/O variance is large enough to make --lite/--full latency
numbers unreliable signal for "which orchestrator is faster" from a
small sample -- they're useful for tool_trace correctness and a
real-world sanity check, not for isolating framework overhead. --pure
is what actually answers that question cleanly.

Fairness note (--lite/--full only): the two orchestrators run
back-to-back against the SAME ticker for a given question, and
yfinance/SEC EDGAR/ChromaDB all cache to disk (filings_cache/,
vector_db/) -- so a naive back-to-back run would make whichever
orchestrator runs second look artificially faster purely from a warm
cache, not real orchestration efficiency. Each question gets one
untimed warm-up call first (see _warm_up) so both timed runs see
identical, already-warm caches.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.agent_constants import TRAILING_TOOLS
from app.agents.langgraph_agent import EVIDENCE_TOOLS, LangGraphResearchAgent
from app.agents.research_agent import ResearchAgent
from app.tools.tool_registry import ToolRegistry

DEFAULT_TICKERS = ["AAPL", "MSFT"]
DEFAULT_PURE_REPS = 20

QUESTION_TEMPLATE = "Should I invest in {ticker}?"

# Fixed plan used for --pure mode -- every evidence tool, so the
# comparison exercises the full routing graph (all 5 conditional
# branches + the 4-tool trailing sequence), not just a 1-tool shortcut.
PURE_MODE_PLAN = list(EVIDENCE_TOOLS)


class _NoOpTool:
    """Stand-in for a real tool -- still records itself in tool_trace
    (so routing/dispatch is exercised exactly as it would be for real)
    but does no actual work. Used for report_tool/evaluation_tool in
    --lite mode, and for every tool in --pure mode."""

    def __init__(self, name):
        self.name = name

    def run(self, context):
        context.record_tool(self.name)
        return context


class _FixedPlanner:
    """Planner stand-in that returns a fixed plan instantly -- removes
    the real LLMPlanner's own LLM-generation call (a real, if small,
    latency cost) from --pure mode, so the only thing left in the
    timed path is orchestration dispatch itself."""

    def __init__(self, plan):
        self._plan = list(plan)

    def create_plan(self, question):
        return list(self._plan)


def _lite_tools():
    """A real ToolRegistry's tools, with report_tool/evaluation_tool
    swapped for no-op stand-ins. Every evidence-gathering tool, plus
    institutional_consensus_tool and news_tool, runs for real."""
    tools = dict(ToolRegistry().tools)
    tools["report_tool"] = _NoOpTool("report_tool")
    tools["evaluation_tool"] = _NoOpTool("evaluation_tool")
    return tools


def _all_stub_tools():
    return {name: _NoOpTool(name) for name in EVIDENCE_TOOLS + TRAILING_TOOLS}


def _warm_up(question: str, mode: str):
    """One untimed call so both timed runs below see warm SEC/yfinance/
    ChromaDB caches -- see module docstring's fairness note. Skipped
    for --pure, which has no real I/O to warm."""
    if mode == "pure":
        return
    tools = _lite_tools() if mode == "lite" else None
    LangGraphResearchAgent(tools=tools).run(question)


def _timed_run(agent, question: str):
    start = time.time()
    context = agent.run(question)
    elapsed = time.time() - start
    return context, elapsed


def _build_agents(mode: str):
    if mode == "pure":
        tools = _all_stub_tools()
        planner = _FixedPlanner(PURE_MODE_PLAN)
    elif mode == "lite":
        tools = _lite_tools()
        planner = None
    else:
        tools = None
        planner = None

    hand_rolled = ResearchAgent()
    if tools is not None:
        hand_rolled.registry.tools = tools
    if planner is not None:
        hand_rolled.planner = planner

    graph_agent = LangGraphResearchAgent(tools=tools, planner=planner)

    return hand_rolled, graph_agent


def run_comparison(tickers, mode: str, pure_reps: int):
    results = []

    for ticker in tickers:
        question = QUESTION_TEMPLATE.format(ticker=ticker)
        print(f"\n{'=' * 70}\n{ticker}: \"{question}\"\n{'=' * 70}")

        print("  warming caches (untimed)..." if mode != "pure" else "  (pure mode -- no warm-up needed)")
        _warm_up(question, mode)

        reps = pure_reps if mode == "pure" else 1
        hand_times, graph_times = [], []
        trace_hand = trace_graph = None

        for _ in range(reps):
            hand_rolled, graph_agent = _build_agents(mode)

            ctx_hand, t_hand = _timed_run(hand_rolled, question)
            ctx_graph, t_graph = _timed_run(graph_agent, question)

            hand_times.append(t_hand)
            graph_times.append(t_graph)
            trace_hand, trace_graph = ctx_hand.tool_trace, ctx_graph.tool_trace

        trace_match = trace_hand == trace_graph
        median_hand = statistics.median(hand_times)
        median_graph = statistics.median(graph_times)

        if mode == "pure":
            print(f"  ResearchAgent:          median {median_hand*1000:.3f}ms over {reps} reps")
            print(f"  LangGraphResearchAgent: median {median_graph*1000:.3f}ms over {reps} reps")
        else:
            print(f"  ResearchAgent:          {median_hand:.2f}s  trace={trace_hand}")
            print(f"  LangGraphResearchAgent: {median_graph:.2f}s  trace={trace_graph}")
        print(f"  tool_trace identical: {trace_match}")
        if not trace_match:
            print("  ** MISMATCH -- the two orchestrators disagreed on which tools to run **")

        results.append({
            "ticker": ticker,
            "hand_rolled_seconds": median_hand,
            "langgraph_seconds": median_graph,
            "trace_match": trace_match,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--lite", action="store_const", dest="mode", const="lite",
        help="6 real tools (network/model calls), report_tool/evaluation_tool stubbed. "
             "Real tool_trace correctness check; NOT a clean latency signal (I/O noise).",
    )
    mode_group.add_argument(
        "--full", action="store_const", dest="mode", const="full",
        help="No stand-ins anywhere -- true end-to-end latency including the LLM call. Slow.",
    )
    parser.add_argument(
        "--pure-reps", type=int, default=DEFAULT_PURE_REPS,
        help=f"Repetitions per ticker in --pure mode (default {DEFAULT_PURE_REPS}).",
    )
    parser.set_defaults(mode="pure")
    args = parser.parse_args()

    results = run_comparison(args.tickers, mode=args.mode, pure_reps=args.pure_reps)

    print(f"\n{'=' * 70}\nSUMMARY ({args.mode} mode)\n{'=' * 70}")
    all_match = all(r["trace_match"] for r in results)
    mean_hand = sum(r["hand_rolled_seconds"] for r in results) / len(results)
    mean_graph = sum(r["langgraph_seconds"] for r in results) / len(results)
    delta_pct = (mean_graph - mean_hand) / mean_hand * 100 if mean_hand else 0.0

    print(f"Queries run: {len(results)}")
    print(f"tool_trace identical on every query: {all_match}")
    if args.mode == "pure":
        # A percentage delta is meaningless (or misleadingly ~0%) when
        # the hand-rolled baseline is itself sub-millisecond -- report
        # the absolute delta instead.
        delta_ms = (mean_graph - mean_hand) * 1000
        print(
            f"Mean of per-ticker medians -- ResearchAgent: {mean_hand*1000:.3f}ms | "
            f"LangGraphResearchAgent: {mean_graph*1000:.3f}ms (+{delta_ms:.3f}ms absolute)"
        )
        print(
            "\nThis isolates pure control-flow dispatch overhead (StateGraph/Pregel traversal "
            "vs. a plain Python for-loop) with all 9 tools stubbed to no-ops -- no network, no "
            "model inference. Run with --lite or --full for real-world numbers (noisier, but "
            "closer to actual user-facing latency)."
        )
    else:
        print(f"Mean latency -- ResearchAgent: {mean_hand:.2f}s | LangGraphResearchAgent: {mean_graph:.2f}s "
              f"({delta_pct:+.1f}%)")
        print(
            "\nNote: real network/model calls make this a noisy measurement of orchestration "
            "overhead specifically -- see the module docstring. Run with --pure for a clean "
            "control-flow-only comparison."
        )


if __name__ == "__main__":
    main()
