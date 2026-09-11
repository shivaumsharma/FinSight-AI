"""
Unit tests for the hand-rolled orchestrator (app/agents/research_agent.py).

ResearchAgent.__init__ builds a real Planner()/ToolRegistry() with no
injection point (unlike langgraph_agent.py's build_graph(tools=,
planner=)) -- tests here construct via __new__ to skip that real
construction entirely and set .planner/.registry to lightweight stubs
directly, so run() itself is exercised with no network/model calls.
resolve_companies is monkeypatched at the module level (same pattern
test_watchlist.py etc. use for main.resolve_companies) since run()
calls it directly, unlike the LangGraph port's tests which invoke the
graph beneath LangGraphResearchAgent.run() and never reach it.
"""

import pytest

from app.agents import research_agent as ra
from app.agents.agent_constants import TRAILING_TOOLS, NoCompanyDetectedError
from app.core.research_context import ResearchContext


class _StubTool:
    def __init__(self, name, raises=False):
        self.name = name
        self.raises = raises

    def run(self, context):
        if self.raises:
            raise RuntimeError(f"{self.name} exploded")
        context.record_tool(self.name)


class _StubPlanner:
    def __init__(self, plan):
        self._plan = list(plan)

    def create_plan(self, question):
        return list(self._plan)


class _StubRegistry:
    def __init__(self, tools):
        self._tools = tools

    def get(self, name):
        return self._tools.get(name)


def _agent(plan, tools):
    agent = ra.ResearchAgent.__new__(ra.ResearchAgent)
    agent.planner = _StubPlanner(plan)
    agent.registry = _StubRegistry(tools)
    return agent


def _stub_tools(**overrides):
    tools = {name: _StubTool(name) for name in ["market_data_tool", "valuation_tool", "rag_tool"] + TRAILING_TOOLS}
    tools.update(overrides)
    return tools


def test_run_executes_planned_tools_in_order_then_trailing_tools(monkeypatch):
    monkeypatch.setattr(ra, "resolve_companies", lambda q: ["AAPL"])
    plan = ["market_data_tool", "valuation_tool", "rag_tool"]
    agent = _agent(plan, _stub_tools())

    context = agent.run("Should I invest in Apple?")

    assert context.tool_trace == plan + TRAILING_TOOLS


def test_run_raises_no_company_detected_when_resolve_companies_finds_nothing(monkeypatch):
    monkeypatch.setattr(ra, "resolve_companies", lambda q: [])
    agent = _agent([], _stub_tools())

    with pytest.raises(NoCompanyDetectedError):
        agent.run("what is the stock market")


def test_run_isolates_a_failing_evidence_tool_and_still_runs_the_rest(monkeypatch):
    # Regression test: a bug in one tool must not crash the whole
    # request and lose every OTHER tool's already-gathered evidence --
    # see run()'s own comment for why. valuation_tool raises; rag_tool
    # and every trailing tool must still run.
    monkeypatch.setattr(ra, "resolve_companies", lambda q: ["AAPL"])
    plan = ["market_data_tool", "valuation_tool", "rag_tool"]
    tools = _stub_tools(valuation_tool=_StubTool("valuation_tool", raises=True))
    agent = _agent(plan, tools)

    context = agent.run("Should I invest in Apple?")

    assert context.tool_trace == ["market_data_tool", "rag_tool"] + TRAILING_TOOLS
    assert context.metadata["tool_errors"] == [{"tool": "valuation_tool", "error": "valuation_tool exploded"}]


def test_run_isolates_a_failing_trailing_tool_and_still_reaches_evaluation(monkeypatch):
    # The trailing tools (report_tool/evaluation_tool) get the same
    # isolation -- a crash in, say, news_tool must not prevent
    # evaluation_tool (always last) from still running and returning a
    # scored context.
    monkeypatch.setattr(ra, "resolve_companies", lambda q: ["AAPL"])
    plan = ["market_data_tool"]
    tools = _stub_tools(news_tool=_StubTool("news_tool", raises=True))
    agent = _agent(plan, tools)

    context = agent.run("Should I invest in Apple?")

    assert "news_tool" not in context.tool_trace
    assert context.tool_trace == ["market_data_tool", "institutional_consensus_tool", "report_tool", "evaluation_tool"]
    assert context.metadata["tool_errors"] == [{"tool": "news_tool", "error": "news_tool exploded"}]


def test_run_records_multiple_failing_tools_independently(monkeypatch):
    monkeypatch.setattr(ra, "resolve_companies", lambda q: ["AAPL"])
    plan = ["market_data_tool", "valuation_tool"]
    tools = _stub_tools(
        market_data_tool=_StubTool("market_data_tool", raises=True),
        valuation_tool=_StubTool("valuation_tool", raises=True),
    )
    agent = _agent(plan, tools)

    context = agent.run("Should I invest in Apple?")

    assert context.tool_trace == TRAILING_TOOLS
    assert context.metadata["tool_errors"] == [
        {"tool": "market_data_tool", "error": "market_data_tool exploded"},
        {"tool": "valuation_tool", "error": "valuation_tool exploded"},
    ]
