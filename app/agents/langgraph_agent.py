"""
langgraph_agent.py

A LangGraph port of ResearchAgent's orchestration, kept alongside the
hand-rolled controller (research_agent.py) as a documented, benchmarked
alternative -- not a replacement. Reuses every piece of business logic
unchanged: company resolution, the Planner (LLM proposal union rule-based
fallback), and every BaseTool subclass. LangGraph is only responsible
for the control flow that ResearchAgent.run()'s `for tool_name in plan`
loop used to own.

Why a routing loop instead of static fan-out edges
----------------------------------------------------
The five evidence-gathering tools all read from and mutate the SAME
`ResearchContext` object, and some have real data dependencies on
others (valuation_tool needs market_data_tool's normalized financials,
sentiment_tool needs rag_tool's retrieved chunks -- both tools already
carry their own "run my dependency first if it hasn't run yet" fallback,
see market_data_tool.py/rag_tool.py). Fanning the planner's chosen tools
out to parallel LangGraph branches would race multiple nodes mutating
that one shared object with no reducer to merge conflicting writes.

Instead, a single `route` node re-examines (state["plan"], state["step"])
after every tool call and conditionally dispatches to the next tool in
the planner's own order, looping back to itself until the plan is
exhausted, then falls through to the fixed trailing sequence
(institutional_consensus_tool -> news_tool -> report_tool ->
evaluation_tool -- the same TRAILING_TOOLS order research_agent.py
always appends). This reproduces the original loop's exact sequential
contract, which is also what makes a fair, tool_trace-identical
benchmark against ResearchAgent possible (see
scripts/benchmark_orchestration.py).

Checkpointing
-------------
build_graph(checkpointer=...) accepts any LangGraph checkpointer (e.g.
langgraph.checkpoint.sqlite.SqliteSaver) to make a run resumable/
inspectable via graph.get_state(config) -- a capability the hand-rolled
controller doesn't have. Left as an opt-in constructor argument (default
None) so the orchestration-latency benchmark isn't skewed by checkpoint
write overhead; see LangGraphResearchAgent.__init__.
"""

from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END

from app.core.research_context import ResearchContext
from app.core.company_resolver import resolve_companies, is_comparison_question
from app.agents.agent_constants import NoCompanyDetectedError, TRAILING_TOOLS
from app.planner import Planner

# ToolRegistry is deliberately NOT imported at module level -- it pulls
# in every tool's own imports (torch, transformers, chromadb,
# llama-cpp-python) just to define 9 classes, even though build_graph()
# only actually needs it when no `tools=` override is supplied. Deferred
# to inside build_graph() so importing this module (and anything that
# always passes tools=, like app/tests/test_langgraph_agent.py) stays
# free of that entire stack -- mirrors the same "defer the expensive
# import" reasoning already used for LocalLlamaProvider's own lazy
# ReportGenerator construction (app/core/llm_provider.py).

# The five tools the planner is allowed to choose between (matches
# LLMPlanner.PLANNABLE_TOOLS plus comparison_tool, which the rule-based
# fallback never proposes on its own -- see planner_rules.py -- but
# ResearchAgent.run() force-appends when context.mode == "comparison").
EVIDENCE_TOOLS = [
    "market_data_tool",
    "valuation_tool",
    "rag_tool",
    "sentiment_tool",
    "comparison_tool",
]


class GraphState(TypedDict):
    context: ResearchContext
    plan: List[str]
    step: int


def _make_resolve_and_plan_node(planner):
    def node(state: GraphState) -> GraphState:
        context = state["context"]
        plan = planner.create_plan(context.question)

        if context.mode == "comparison" and "comparison_tool" not in plan:
            plan.append("comparison_tool")

        context.add_metadata("plan", plan)
        state["plan"] = plan
        state["step"] = 0
        return state
    return node


def _route_passthrough_node(state: GraphState) -> GraphState:
    """No-op node that exists purely as an attachment point for the
    conditional edge below -- add_conditional_edges routes FROM a node,
    so the dispatch decision needs somewhere to live that every evidence
    tool can loop back to."""
    return state


def _evidence_dispatch(state: GraphState) -> str:
    """Router: name of the next evidence tool to run, in the planner's
    own order, or 'institutional_consensus_tool' (the first trailing
    tool) once the plan is exhausted -- mirrors
    `for tool_name in plan: tool.run(context)` one step at a time."""
    plan = state["plan"]
    step = state["step"]
    if step >= len(plan):
        return "institutional_consensus_tool"
    return plan[step]


def _make_evidence_node(tool):
    def node(state: GraphState) -> GraphState:
        tool.run(state["context"])
        state["step"] += 1
        return state
    return node


def _make_trailing_node(tool):
    def node(state: GraphState) -> GraphState:
        tool.run(state["context"])
        return state
    return node


def build_graph(checkpointer=None, tools=None, planner=None):
    """Builds and compiles the StateGraph. Exposed standalone (not just
    via LangGraphResearchAgent) so scripts/benchmark_orchestration.py
    and tests can construct it directly, and so a caller can pass a
    real checkpointer (e.g. SqliteSaver) for resumable runs without
    LangGraphResearchAgent needing to know about every checkpointer
    implementation.

    tools/planner: optional injected {name: BaseTool instance} map and
    Planner-shaped object (anything with .create_plan(question) -> list[str]),
    defaulting to a real ToolRegistry()'s tools and a real Planner() --
    overridden in tests (app/tests/test_langgraph_agent.py) with
    lightweight stubs so routing/dispatch logic can be verified without
    any network or model calls, the same dependency-injection pattern
    LLMPlanner already uses for its generator.
    """

    if tools is None:
        from app.tools.tool_registry import ToolRegistry
        tools = ToolRegistry().tools
    planner = planner if planner is not None else Planner()
    graph = StateGraph(GraphState)

    graph.add_node("resolve_and_plan", _make_resolve_and_plan_node(planner))
    graph.add_node("route", _route_passthrough_node)

    for name in EVIDENCE_TOOLS:
        graph.add_node(name, _make_evidence_node(tools[name]))
        graph.add_edge(name, "route")

    for name in TRAILING_TOOLS:
        graph.add_node(name, _make_trailing_node(tools[name]))

    graph.add_edge(START, "resolve_and_plan")
    graph.add_edge("resolve_and_plan", "route")

    path_map = {name: name for name in EVIDENCE_TOOLS}
    path_map["institutional_consensus_tool"] = "institutional_consensus_tool"
    graph.add_conditional_edges("route", _evidence_dispatch, path_map)

    # TRAILING_TOOLS is ["institutional_consensus_tool", "news_tool",
    # "report_tool", "evaluation_tool"] -- chain them in that fixed
    # order, same as ResearchAgent.run()'s trailing-tools loop.
    for earlier, later in zip(TRAILING_TOOLS, TRAILING_TOOLS[1:]):
        graph.add_edge(earlier, later)
    graph.add_edge(TRAILING_TOOLS[-1], END)

    return graph.compile(checkpointer=checkpointer)


class LangGraphResearchAgent:
    """
    Drop-in alternative to ResearchAgent with the same public contract
    (`run(question) -> ResearchContext`, raises NoCompanyDetectedError
    for the same reason) -- see research_agent.py's module docstring
    for what the hand-rolled version does; this does the identical
    thing, routed through a compiled LangGraph graph instead of a plain
    Python for-loop.
    """

    def __init__(self, checkpointer=None, tools=None, planner=None):
        # tools/planner: same injection points as build_graph() -- used
        # by scripts/benchmark_orchestration.py's "lite" mode to swap in
        # no-op report_tool/evaluation_tool stand-ins so the benchmark
        # measures orchestration overhead, not LLM inference time.
        self.graph = build_graph(checkpointer=checkpointer, tools=tools, planner=planner)

    def run(self, question: str) -> ResearchContext:

        companies = resolve_companies(question)

        if not companies:
            raise NoCompanyDetectedError(question)

        context = ResearchContext(
            ticker=companies[0],
            question=question,
        )

        if is_comparison_question(question) and len(companies) >= 2:
            context.mode = "comparison"
            context.metadata["peer_ticker"] = companies[1]

        initial_state: GraphState = {"context": context, "plan": [], "step": 0}
        final_state = self.graph.invoke(initial_state)

        return final_state["context"]
