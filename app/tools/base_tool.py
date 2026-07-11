"""
base_tool.py

Abstract base class for every tool available to the ResearchAgent.

Design decision
----------------
The original BaseTool used `run(self, **kwargs) -> Any`, and every
tool re-implemented its own ad-hoc return dict. That made it
impossible for one tool to build on another tool's output without
the caller manually threading kwargs between calls.

The project already has a single shared state object,
`ResearchContext`, used everywhere else in the pipeline
(EvidenceBuilder, ReportGenerationEngine, EvaluationEngine,
ResearchLogger). Tools now follow the same contract:

    tool.run(context: ResearchContext) -> ResearchContext

Every tool reads whatever fields it needs from the context and
writes its results back onto the same object. This means:

- Tools can be freely reordered/composed by the planner.
- A later tool (e.g. ValuationTool) can reuse data a previous tool
  already fetched (e.g. MarketDataTool) instead of re-fetching it.
- The exact same context object can be hand off to the existing
  report/evaluation/logging code with zero adapters.
"""

from abc import ABC, abstractmethod

from app.core.research_context import ResearchContext


class BaseTool(ABC):
    """
    Abstract base class for every tool.

    Subclasses must set `name` and `description`. `description` is
    shown to the LLM Planner so it can decide when to invoke the
    tool, so it should be a short, precise, one-line explanation of
    what the tool does and what it needs/produces.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, context: ResearchContext) -> ResearchContext:
        """
        Execute the tool against the shared research context and
        return the (mutated) context.
        """
        raise NotImplementedError
