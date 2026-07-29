"""
Unit tests for the LangGraph orchestrator (app/agents/langgraph_agent.py).

Uses injected stub tools/planner (build_graph's tools=/planner= params)
so these run in milliseconds with no network or model calls -- the same
dependency-injection pattern LLMPlanner already uses for its generator.
What's under test is the *routing* logic (does the graph visit the
right nodes, in the right order, for a given plan) -- not any real
tool's behavior, which is already covered by each tool's own tests.
"""

import pytest

from app.agents.langgraph_agent import (
    EVIDENCE_TOOLS, GraphState, build_graph, _evidence_dispatch,
)
from app.agents.agent_constants import TRAILING_TOOLS
from app.core.research_context import ResearchContext


class _StubTool:
    def __init__(self, name):
        self.name = name

    def run(self, context):
        context.record_tool(self.name)
        return context


class _StubPlanner:
    def __init__(self, plan):
        self._plan = list(plan)

    def create_plan(self, question):
        return list(self._plan)


def _stub_tools():
    return {name: _StubTool(name) for name in EVIDENCE_TOOLS + TRAILING_TOOLS}


# ---------------------------------------------------------- _evidence_dispatch

def test_evidence_dispatch_returns_next_step_in_plan():
    state: GraphState = {"plan": ["rag_tool", "sentiment_tool"], "step": 0, "context": None}
    assert _evidence_dispatch(state) == "rag_tool"


def test_evidence_dispatch_advances_with_step():
    state: GraphState = {"plan": ["rag_tool", "sentiment_tool"], "step": 1, "context": None}
    assert _evidence_dispatch(state) == "sentiment_tool"


def test_evidence_dispatch_falls_through_to_trailing_tools_once_plan_exhausted():
    state: GraphState = {"plan": ["rag_tool"], "step": 1, "context": None}
    assert _evidence_dispatch(state) == "institutional_consensus_tool"


def test_evidence_dispatch_handles_empty_plan():
    state: GraphState = {"plan": [], "step": 0, "context": None}
    assert _evidence_dispatch(state) == "institutional_consensus_tool"


# ---------------------------------------------------------- full graph runs

def test_graph_compiles_with_every_expected_node():
    graph = build_graph(tools=_stub_tools(), planner=_StubPlanner([]))
    nodes = set(graph.get_graph().nodes.keys())
    for name in EVIDENCE_TOOLS + TRAILING_TOOLS:
        assert name in nodes
    assert "resolve_and_plan" in nodes
    assert "route" in nodes


def test_graph_visits_only_planned_evidence_tools_then_trailing_tools():
    plan = ["market_data_tool", "valuation_tool"]
    graph = build_graph(tools=_stub_tools(), planner=_StubPlanner(plan))

    context = ResearchContext(ticker="AAPL", question="What is Apple's intrinsic value?")
    final_state = graph.invoke({"context": context, "plan": [], "step": 0})

    trace = final_state["context"].tool_trace
    assert trace == plan + TRAILING_TOOLS


def test_graph_skips_evidence_tools_not_in_the_plan():
    # A pure "summarize the earnings call" plan should never touch
    # valuation_tool/sentiment_tool/comparison_tool.
    plan = ["rag_tool"]
    graph = build_graph(tools=_stub_tools(), planner=_StubPlanner(plan))

    context = ResearchContext(ticker="AAPL", question="What did management say about AI?")
    final_state = graph.invoke({"context": context, "plan": [], "step": 0})

    trace = final_state["context"].tool_trace
    assert "valuation_tool" not in trace
    assert "sentiment_tool" not in trace
    assert "comparison_tool" not in trace
    assert trace == plan + TRAILING_TOOLS


def test_graph_handles_a_full_five_tool_plan_in_order():
    plan = list(EVIDENCE_TOOLS)  # every evidence tool, in the planner's own order
    graph = build_graph(tools=_stub_tools(), planner=_StubPlanner(plan))

    context = ResearchContext(ticker="AAPL", question="Should I invest in Apple?")
    context.mode = "comparison"
    context.metadata["peer_ticker"] = "MSFT"
    final_state = graph.invoke({"context": context, "plan": [], "step": 0})

    trace = final_state["context"].tool_trace
    assert trace == plan + TRAILING_TOOLS


def test_graph_appends_comparison_tool_in_comparison_mode_even_if_planner_omits_it():
    # Mirrors ResearchAgent.run()'s own force-append: "if context.mode ==
    # 'comparison' and 'comparison_tool' not in plan: plan.append(...)".
    plan = ["market_data_tool", "valuation_tool"]
    graph = build_graph(tools=_stub_tools(), planner=_StubPlanner(plan))

    context = ResearchContext(ticker="AAPL", question="Compare Apple and Microsoft")
    context.mode = "comparison"
    context.metadata["peer_ticker"] = "MSFT"
    final_state = graph.invoke({"context": context, "plan": [], "step": 0})

    assert "comparison_tool" in final_state["context"].tool_trace
    assert final_state["context"].metadata["plan"][-1] == "comparison_tool"


def test_graph_tool_trace_matches_research_agent_for_the_same_plan():
    """
    The core claim behind the LangGraph port: for an identical planner
    decision, the graph produces the exact same tool execution order as
    ResearchAgent.run()'s plain `for tool_name in plan: tool.run(context)`
    loop plus its trailing-tools append -- see research_agent.py.
    """
    plan = ["market_data_tool", "valuation_tool", "rag_tool"]

    # What ResearchAgent.run() would do for this plan (no LLM/network
    # involved -- just the loop's own logic, reproduced directly).
    expected_trace = list(plan)
    for tool_name in TRAILING_TOOLS:
        if tool_name not in expected_trace:
            expected_trace.append(tool_name)

    graph = build_graph(tools=_stub_tools(), planner=_StubPlanner(plan))
    context = ResearchContext(ticker="AAPL", question="Should I invest in Apple?")
    final_state = graph.invoke({"context": context, "plan": [], "step": 0})

    assert final_state["context"].tool_trace == expected_trace
