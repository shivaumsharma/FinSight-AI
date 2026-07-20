"""
valuation_tool.py

Runs the DCF valuation pipeline (FCFF -> WACC -> DCF -> Sensitivity),
reusing the existing, unmodified valuation engines in app/valuation/.

Previously this tool re-fetched market data and re-normalized
financial statements itself, duplicating exactly what
CompanyTool/FinancialTool already did. It now depends on
MarketDataTool having already populated `context.normalized_financials`
/ `context.market_cap` / `context.beta`, and will transparently run
MarketDataTool first if that hasn't happened yet (e.g. if a caller
invokes ValuationTool directly, or the planner picks it without
market_data_tool for some reason).
"""

from app.core.research_context import ResearchContext
from app.valuation.valuation_pipeline import ValuationPipeline
from app.valuation.valuation_summary import ValuationSummaryBuilder
from app.valuation.relative_valuation import RelativeValuationEngine
from app.valuation.monte_carlo_dcf import MonteCarloDCFEngine
from app.valuation.ml_features import extract_features
from app.valuation.ml_valuation_classifier import predict_verdict
from .base_tool import BaseTool


class ValuationTool(BaseTool):

    name = "valuation_tool"
    description = (
        "Runs a full DCF valuation (FCFF forecast, WACC, enterprise value, "
        "equity value, intrinsic value per share, sensitivity matrix). "
        "Required for valuation/intrinsic-value/undervalued/overvalued questions. "
        "Depends on market_data_tool having already run for this ticker."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        if context.normalized_financials is None:
            from .market_data_tool import MarketDataTool
            MarketDataTool().run(context)

        pipeline = ValuationPipeline(
            financial_df=context.normalized_financials,
            market_cap=context.market_cap,
            beta=context.beta,
        )

        results = pipeline.run_valuation()

        shares_outstanding = None
        if "shares_outstanding" in context.normalized_financials.columns:
            series = context.normalized_financials["shares_outstanding"].dropna()
            if not series.empty:
                shares_outstanding = series.iloc[-1]

        if shares_outstanding and results.get("dcf_available"):
            results["intrinsic_value"] = results["equity_value"] / shares_outstanding

        current_price = (context.company_info or {}).get("current_price")
        if current_price:
            results["current_price"] = current_price
            if results.get("intrinsic_value"):
                results["upside_percent"] = round(
                    (results["intrinsic_value"] - current_price) / current_price * 100,
                    2,
                )

        results["relative_valuation"] = RelativeValuationEngine(
            financial_df=context.normalized_financials,
            historical_prices=context.historical_prices,
            market_cap=context.market_cap,
            current_price=current_price,
        ).evaluate()

        # Monte Carlo distribution around the DCF point estimate --
        # see monte_carlo_dcf.py. Statistics (percentiles, prob of
        # undervaluation) need current_price, which isn't known inside
        # ValuationPipeline, so the raw sampled values are computed
        # there and turned into statistics here.
        mc_values = results.pop("monte_carlo_values", None)
        results["monte_carlo"] = (
            MonteCarloDCFEngine.statistics(mc_values, current_price)
            if mc_values is not None and current_price
            else None
        )

        context.valuation_results = results
        context.enterprise_value = results.get("enterprise_value")
        context.equity_value = results.get("equity_value")
        context.intrinsic_value = results.get("intrinsic_value")

        # ML valuation classifier -- see ml_valuation_classifier.py.
        # Display-only, NOT folded into the recommendation composite
        # (report_data_builder.py's DCF_WEIGHT/RELATIVE_WEIGHT) --
        # this signal has no accuracy track record yet. None if no
        # trained model exists (scripts/train_ml_classifier.py hasn't
        # been run) or if DCF was unavailable for this company (see
        # extract_features).
        ml_features = extract_features(context)
        results["ml_classifier"] = predict_verdict(ml_features) if ml_features else None

        context.valuation_summary = ValuationSummaryBuilder().build(results)

        context.record_tool(self.name)

        return context
