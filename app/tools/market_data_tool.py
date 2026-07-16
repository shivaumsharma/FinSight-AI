"""
market_data_tool.py

Consolidates the old CompanyTool + FinancialTool + MarketTool.

Why merge three tools into one
-------------------------------
All three old tools independently constructed their own
`MarketDataLoader(ticker)` and hit the yfinance API separately, even
though they were almost always needed together (ValuationTool then
did the *same* fetch a fourth time). That's four network round-trips
for data that only needs to be fetched once per ticker per request.

MarketDataTool fetches company profile, financial statements, and
historical prices in one pass, normalizes the statements (reusing
`FinancialStatementNormaliser` unchanged) and writes everything onto
`ResearchContext`. Every downstream tool (ValuationTool,
ComparisonTool, ReportTool) reads from the context instead of
re-fetching.
"""

from app.core.research_context import ResearchContext
from app.data.market_data import MarketDataLoader, TickerNotFoundError
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.analysis.financial_analysis import FinancialAnalysisBuilder
from .base_tool import BaseTool


class MarketDataTool(BaseTool):

    name = "market_data_tool"
    description = (
        "Fetches company profile, financial statements (income statement, "
        "balance sheet, cash flow) and historical prices for a ticker, then "
        "normalizes and summarizes the financials. Required as the first "
        "step before valuation, financial-metric, or investment-recommendation "
        "questions."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        loader = MarketDataLoader(context.ticker)

        context.company_info = loader.get_company_info()

        # yfinance doesn't raise for an invalid/mistyped/delisted
        # ticker -- it just returns an info dict with nothing useful
        # in it. Catching that here, before any statement fetch, means
        # the user sees one clear "ticker not found" message instead
        # of whichever of get_income_statement/get_balance_sheet/
        # get_cash_flow happens to hit an empty DataFrame first.
        if not context.company_info.get("company_name") and not context.company_info.get("current_price"):
            raise TickerNotFoundError(context.ticker)

        context.income_statement = loader.get_income_statement()
        context.balance_sheet = loader.get_balance_sheet()
        context.cash_flow = loader.get_cash_flow()

        # Historical prices are only needed for a handful of question
        # types (e.g. "how has the stock performed"); fetch them but
        # never let a missing price history break the rest of the run.
        try:
            context.historical_prices = loader.get_historical_prices()
        except Exception:
            context.historical_prices = None

        context.market_cap = context.company_info.get("market_cap")

        # Prefer the real beta from yfinance; fall back to the
        # project's previous hardcoded market-average assumption of
        # 1.2 only when the data provider doesn't report one.
        context.beta = context.company_info.get("beta") or 1.2

        normaliser = FinancialStatementNormaliser(
            context.income_statement,
            context.balance_sheet,
            context.cash_flow,
        )

        context.normalized_financials = normaliser.normalise()

        summary_builder = FinancialAnalysisBuilder()
        context.financial_summary = summary_builder.build(context.normalized_financials)
        context.financial_summary.update(
            summary_builder.interpret(context.financial_summary)
        )

        context.record_tool(self.name)

        return context
