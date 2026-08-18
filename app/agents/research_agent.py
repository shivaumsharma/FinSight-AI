"""
research_agent.py

Orchestrates plan execution against a single, shared ResearchContext.

Previously ResearchAgent.run() took **kwargs, asked the (broken)
rule-based Planner for a list of tool names, and called
`tool.run(**kwargs)` for each -- but every real tool needed a
different subset of kwargs, tools couldn't see each other's output,
and ToolRegistry() itself raised NameError on construction
(ThisisTool), so this class could never actually run.

The rewritten agent:
 1. Resolves the company/companies mentioned in the question
    deterministically (see company_resolver.py) -- NOT via a sidebar
    ticker default (there isn't one anymore: FinSight only researches
    companies actually named in the question) and NOT via the LLM,
    and for comparison questions, stashes the peer ticker in
    context.metadata for ComparisonTool.
 2. Builds one ResearchContext for the request.
 3. Asks the Planner (LLM-first, rule-based fallback) which
    evidence-gathering tools are needed.
 4. Executes each planned tool against the context in order.
 5. Always appends report_tool then evaluation_tool, regardless of
    what the planner chose -- every request should end with an
    answer and a score, and there is no reason to spend planner
    reasoning on a decision that has only one sane answer.
"""

from typing import Callable, Optional

from app.core.research_context import ResearchContext
from app.core.company_resolver import resolve_companies, is_comparison_question
from app.planner import Planner
from app.tools.tool_registry import ToolRegistry
from app.agents.agent_constants import TRAILING_TOOLS, NoCompanyDetectedError


class ResearchAgent:
    """
    Orchestrates the execution of tools against a ResearchContext.

    Contains NO financial logic itself -- it asks the planner what to
    do, executes the tools in order, and returns the fully populated
    context.
    """

    def __init__(self):
        self.planner = Planner()
        self.registry = ToolRegistry()

    def run(
        self,
        question: str,
        risk_tolerance: str = "Moderate",
        on_step: Optional[Callable[[str], None]] = None,
    ) -> ResearchContext:

        companies = resolve_companies(question)

        if not companies:
            raise NoCompanyDetectedError(question)

        context = ResearchContext(
            ticker=companies[0],
            question=question,
            risk_tolerance=risk_tolerance,
        )

        if is_comparison_question(question) and len(companies) >= 2:
            context.mode = "comparison"
            context.metadata["peer_ticker"] = companies[1]

        plan = self.planner.create_plan(question)

        if context.mode == "comparison" and "comparison_tool" not in plan:
            plan.append("comparison_tool")

        for tool_name in TRAILING_TOOLS:
            if tool_name not in plan:
                plan.append(tool_name)

        context.add_metadata("plan", plan)

        # Sentinel fired once, before any tool runs, so the caller (see
        # app/api/jobs.py's _run_job/_on_step) learns this question's
        # REAL, full step list up front -- the plan is dynamic per
        # question (see this module's docstring), so there is no fixed
        # list a caller could otherwise know in advance.
        if on_step:
            on_step("__plan__:" + ",".join(plan))

        for tool_name in plan:
            tool = self.registry.get(tool_name)
            if tool is None:
                continue
            if on_step:
                on_step(tool_name)
            tool.run(context)

        return context
