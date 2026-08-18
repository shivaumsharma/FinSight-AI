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
from app.data.crypto_resolver import is_crypto_ticker
from app.valuation.valuation_pipeline import ValuationPipeline
from app.valuation.valuation_summary import ValuationSummaryBuilder
from app.valuation.relative_valuation import RelativeValuationEngine
from app.valuation.monte_carlo_dcf import MonteCarloDCFEngine
from app.valuation.ml_features import extract_features
from app.valuation.ml_valuation_classifier import predict_verdict
from app.analysis.alpha_factors import AlphaFactorsEngine
from app.data.market_data import get_benchmark_history, SECTOR_ETF_PROXIES
from .base_tool import BaseTool

# Shared shape with ValuationPipeline._unavailable_result -- every
# numeric field explicitly None, dcf_available False, so every caller
# already built around "DCF unavailable" (report_data_builder.py's
# derive_recommendation, stock_score.py's star scoring) handles a
# crypto ticker the exact same way it already handles a real equity
# with structurally missing financials. Not fabricated: no financial
# statements exist for crypto (see market_data_tool.py's guard), so
# there is genuinely nothing to compute here.
_CRYPTO_UNAVAILABLE_REASON = "No financial statements exist for cryptocurrencies -- DCF valuation is not applicable."


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

        if is_crypto_ticker(context.ticker):
            current_price = (context.company_info or {}).get("current_price")
            results = {
                "dcf_available": False,
                "dcf_unavailable_reason": _CRYPTO_UNAVAILABLE_REASON,
                "enterprise_value": None, "equity_value": None, "intrinsic_value": None,
                "fcff_forecasts": None, "wacc": None, "raw_wacc": None, "wacc_floored": False,
                "wacc_floor_note": None, "terminal_growth_rate": None, "sensitivity_analysis": None,
                "current_price": current_price, "upside_percent": None,
                "relative_valuation": None, "monte_carlo": None, "ml_classifier": None,
                "alpha_factors": {},
                "is_crypto": True,
            }
            context.valuation_results = results
            context.enterprise_value = None
            context.equity_value = None
            context.intrinsic_value = None
            context.valuation_summary = ValuationSummaryBuilder().build(results)
            context.record_tool(self.name)
            return context

        pipeline = ValuationPipeline(
            financial_df=context.normalized_financials,
            market_cap=context.market_cap,
            beta=context.beta,
            ticker=context.ticker,
            currency=(context.company_info or {}).get("currency"),
            risk_tolerance=context.risk_tolerance,
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

        # Alpha Factors scorecard -- see alpha_factors.py's own module
        # docstring. Same non-negotiable boundary as ml_classifier
        # above: display-only, never read by report_data_builder.py's
        # composite score. The three benchmark-comparison factors
        # (Relative Strength vs Index, Sector Relative Performance,
        # Interest Rate Sensitivity) are skipped entirely for .NS
        # tickers -- comparing a rupee-denominated stock against a
        # USD-denominated benchmark isn't a meaningful signal (see
        # AlphaFactorsEngine's own docstring) -- so those fetches are
        # never even made for an NSE ticker.
        is_non_us_listing = (context.ticker or "").upper().endswith(".NS")
        benchmark_history = None
        sector_history = None
        rate_proxy_history = None
        if not is_non_us_listing:
            benchmark_history = get_benchmark_history("^GSPC")
            rate_proxy_history = get_benchmark_history("^TNX")
            sector_etf = SECTOR_ETF_PROXIES.get((context.company_info or {}).get("sector"))
            if sector_etf:
                sector_history = get_benchmark_history(sector_etf)

        results["alpha_factors"] = AlphaFactorsEngine(
            normalized_financials=context.normalized_financials,
            historical_prices=context.historical_prices,
            beta=context.beta,
            company_info=context.company_info,
            # financial_summary/sentiment_summary default to "" (an
            # empty string, not a dict) on a fresh ResearchContext until
            # their own tools populate them -- see research_context.py's
            # dataclass fields. Guard rather than assume dict.
            financial_summary=context.financial_summary if isinstance(context.financial_summary, dict) else {},
            sentiment_summary=context.sentiment_summary if isinstance(context.sentiment_summary, dict) else {},
            news_sentiment_summary=context.news_sentiment_summary or {},
            is_non_us_listing=is_non_us_listing,
            benchmark_history=benchmark_history,
            sector_history=sector_history,
            rate_proxy_history=rate_proxy_history,
        ).evaluate()

        context.valuation_summary = ValuationSummaryBuilder().build(results)

        context.record_tool(self.name)

        return context
