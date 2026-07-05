from .market_tool import MarketTool
from .financial_tool import FinancialTool
from .company_tool import CompanyTool
from .valuation_tool import ValuationTool
from .rag_tool import RAGTool
from .sentiment_tool import SentimentTool
from .thesis_tool import ThesisTool


class ToolRegistry:
    """
    Stores and manages all tools available to the ResearchAgent.
    """

    def __init__(self):
        self.tools = {}

        self.register(MarketTool())
        self.register(FinancialTool())
        self.register(CompanyTool())
        self.register(ValuationTool())
        self.register(RAGTool())
        self.register(SentimentTool())
        self.register(ThesisTool())

    def register(self, tool):
        self.tools[tool.name] = tool

    def get(self, tool_name):
        return self.tools.get(tool_name)

    def list_tools(self):
        return list(self.tools.keys())