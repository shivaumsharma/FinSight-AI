from app.data.market_data import MarketDataLoader
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.valuation.valuation_pipeline import ValuationPipeline
from .base_tool import BaseTool


class ValuationTool(BaseTool):

    name = "valuation_tool"
    description = "Runs DCF valuation."

    def run(self, ticker: str, **kwargs):

        loader = MarketDataLoader(ticker)

        income = loader.get_income_statement()
        balance = loader.get_balance_sheet()
        cashflow = loader.get_cash_flow()

        normaliser = FinancialStatementNormaliser(
            income,
            balance,
            cashflow
        )

        financial_df = normaliser.normalise()

        company_info = loader.get_company_info()

        market_cap = company_info["market_cap"]

        beta = 1.2

        pipeline = ValuationPipeline(
            financial_df=financial_df,
            market_cap=market_cap,
            beta=beta
        )

        return pipeline.run_valuation()