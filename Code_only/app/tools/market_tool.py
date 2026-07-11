from app.data.market_data import MarketDataLoader
from .base_tool import BaseTool


class MarketTool(BaseTool):

    name = "market_tool"
    description = "Fetches historical market price data."

    def run(self, ticker: str, period: str = "5y", **kwargs):

        loader = MarketDataLoader(ticker)

        return loader.get_historical_prices(period)