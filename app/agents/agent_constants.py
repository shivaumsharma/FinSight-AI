"""
agent_constants.py

TRAILING_TOOLS and NoCompanyDetectedError, shared by both orchestrators
(research_agent.py's hand-rolled controller and langgraph_agent.py's
StateGraph port) so neither has to duplicate them or import the other's
heavier module just to reuse two small, dependency-free definitions.
Deliberately has zero imports beyond the stdlib -- anything that imports
this file (including test modules) should never transitively pull in
ToolRegistry's full model/network stack (torch, transformers, chromadb,
llama-cpp-python) just to see this list or catch this exception.
"""

TRAILING_TOOLS = ["institutional_consensus_tool", "news_tool", "report_tool", "evaluation_tool"]


class NoCompanyDetectedError(Exception):
    """
    Raised when no publicly listed company can be identified in the
    question. FinSight only performs company/investment research --
    it never falls back to a default ticker or attempts to answer a
    general finance question. The UI is expected to check
    resolve_companies() itself before ever calling an orchestrator, so
    this mainly guards other/future callers that skip that check.
    """
