from app.data.market_data import MarketDataLoader
from .base_tool import BaseTool


class FinancialTool(BaseTool):

    name = "financial_tool"
    description = "Fetches company financial statements."

    def run(self, ticker: str, **kwargs):

        loader = MarketDataLoader(ticker)

        return {
            "income_statement": loader.get_income_statement(),
            "balance_sheet": loader.get_balance_sheet(),
            "cash_flow": loader.get_cash_flow()
        }