"""
tool_registry.py

Stores and manages all tools available to the ResearchAgent /
LLMPlanner. The previous registry hard-registered a `ThesisTool`
that was never defined anywhere in the codebase -- constructing
`ToolRegistry()` raised `NameError` immediately, meaning the entire
agents/planner/tools package was dead code that could not have run.
That bug is fixed by only registering tools that actually exist.
"""

from .market_data_tool import MarketDataTool
from .valuation_tool import ValuationTool
from .rag_tool import RAGTool
from .sentiment_tool import SentimentTool
from .comparison_tool import ComparisonTool
from .report_tool import ReportTool
from .evaluation_tool import EvaluationTool


class ToolRegistry:
    """
    Stores and manages all tools available to the ResearchAgent.
    """

    def __init__(self):
        self.tools = {}

        for tool_cls in (
            MarketDataTool,
            ValuationTool,
            RAGTool,
            SentimentTool,
            ComparisonTool,
            ReportTool,
            EvaluationTool,
        ):
            self.register(tool_cls())

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, tool_name):
        return self.tools.get(tool_name)

    def list_tools(self):
        return list(self.tools.keys())

    def tool_descriptions(self):
        """Returns {tool_name: description} -- fed to the LLM Planner's prompt."""
        return {name: tool.description for name, tool in self.tools.items()}
