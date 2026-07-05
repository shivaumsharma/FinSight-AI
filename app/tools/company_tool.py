from app.data.market_data import MarketDataLoader
from .base_tool import BaseTool


class CompanyTool(BaseTool):

    name = "company_tool"
    description = "Fetches company profile information."

    def run(self, ticker: str, **kwargs):

        loader = MarketDataLoader(ticker)

        return loader.get_company_info()