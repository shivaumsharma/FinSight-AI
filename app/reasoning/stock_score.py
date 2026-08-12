"""
stock_score.py

Powers the stock-detail-page's "Performance Dashboard" (star-rating
summary) and "AI Insights" card -- see GET /v1/stocks/{ticker}/insights
in app/api/main.py.

Deliberately LLM-free and RAG-free -- this is a lightweight dashboard
widget a user sees on every visit to a ticker's page, not the full,
expensive "Run Full Research Report" flow (which still exists
unchanged, reachable from this same page). Every input comes from the
same deterministic tool chain the full pipeline already uses
(MarketDataTool -> ValuationTool -> derive_recommendation), just called
directly on a bare ResearchContext instead of through ResearchAgent --
the exact same "standalone tool" pattern scripts/phase2_backtest.py
already established for point-in-time backtesting. "AI Insights" here
means algorithmically-derived flags from these signals via disclosed
thresholds, not an LLM-generated narrative (that already exists in the
full report's Investment Thesis section).

Category star scores (1-5) each degrade independently to None when
their underlying data isn't available -- same "missing input blanks
out only that one factor" convention as AlphaFactorsEngine itself.
Ownership is deliberately not a category: no shareholding-pattern data
source exists anywhere in this app (see the approved stock-detail-page
plan's "Explicitly out of scope" section).

Risk is scored so that MORE stars = SAFER (lower risk), not "more
risk" -- consistent with every other category here where more stars is
always the more favorable reading.
"""

from typing import Optional

from app.core.research_context import ResearchContext
from app.reporting.consensus_score import build_consensus_report
from app.reporting.institutional_ratings import fetch_institutional_ratings
from app.reporting.report_data_builder import derive_recommendation
from app.tools.valuation_tool import ValuationTool

STAR_MIN, STAR_MAX = 1, 5


def _clip_star(value: float) -> int:
    return max(STAR_MIN, min(STAR_MAX, round(value)))


def _threshold_star(value: Optional[float], breakpoints: list) -> Optional[int]:
    """breakpoints: descending list of (floor, star) pairs, e.g.
    [(20, 5), (10, 4), (5, 3), (0, 2)] with an implicit final star=1
    below the last floor. None in, None out."""
    if value is None:
        return None
    for floor, star in breakpoints:
        if value >= floor:
            return star
    return 1


def _average_stars(stars: list) -> Optional[int]:
    present = [s for s in stars if s is not None]
    if not present:
        return None
    return _clip_star(sum(present) / len(present))


def _valuation_star(recommendation: dict) -> Optional[int]:
    # composite_score is already a -100..+100 percentile-based read (see
    # report_data_builder.py's _composite_score) -- a linear rescale
    # onto 1-5 stars, not a second independent judgment of cheapness.
    composite = recommendation.get("composite_score")
    if composite is None:
        return None
    return _clip_star((composite + 100) / 200 * 4 + 1)


def _financial_health_star(quality: dict) -> Optional[int]:
    debt_to_equity = quality.get("Debt to Equity")
    current_ratio = quality.get("Current Ratio")
    interest_coverage = quality.get("Interest Coverage Ratio")

    stars = [
        _threshold_star(-debt_to_equity if debt_to_equity is not None else None, [(-0.5, 5), (-1.0, 4), (-2.0, 3), (-3.0, 2)]),
        _threshold_star(current_ratio, [(2.0, 5), (1.5, 4), (1.0, 3), (0.5, 2)]),
        _threshold_star(interest_coverage, [(8.0, 5), (4.0, 4), (2.0, 3), (1.0, 2)]),
    ]
    return _average_stars(stars)


def _growth_star(financial: dict) -> Optional[int]:
    revenue_cagr = financial.get("Revenue CAGR (%)")
    return _threshold_star(revenue_cagr, [(15.0, 5), (8.0, 4), (3.0, 3), (0.0, 2)])


def _profitability_star(quality: dict, financial: dict) -> Optional[int]:
    roe = quality.get("Return on Equity (%)")
    net_margin = financial.get("Net Margin (%)")
    stars = [
        _threshold_star(roe, [(20.0, 5), (15.0, 4), (10.0, 3), (5.0, 2)]),
        _threshold_star(net_margin, [(20.0, 5), (10.0, 4), (5.0, 3), (0.0, 2)]),
    ]
    return _average_stars(stars)


def _momentum_star(market: dict) -> Optional[int]:
    six_month = market.get("6-Month Price Momentum (%)")
    twelve_month = market.get("12-Month Price Momentum (%)")
    stars = [
        _threshold_star(six_month, [(20.0, 5), (10.0, 4), (0.0, 3), (-10.0, 2)]),
        _threshold_star(twelve_month, [(20.0, 5), (10.0, 4), (0.0, 3), (-10.0, 2)]),
    ]
    return _average_stars(stars)


_ALTMAN_ZONE_STARS = {"Safe Zone": 5, "Grey Zone": 3, "Distress Zone": 1}


def _risk_star(risk: dict) -> Optional[int]:
    zone = risk.get("Altman Z-Score Zone")
    volatility = risk.get("Annualized Volatility (%)")
    stars = [
        _ALTMAN_ZONE_STARS.get(zone),
        # Lower volatility = safer = more stars, hence the negation
        # (the same descending-breakpoint helper as everywhere else,
        # applied to -volatility rather than a bespoke ascending one).
        _threshold_star(-volatility if volatility is not None else None, [(-20.0, 5), (-30.0, 4), (-40.0, 3), (-60.0, 2)]),
    ]
    return _average_stars(stars)


def _overall_score(category_scores: dict) -> Optional[int]:
    stars = [v for v in category_scores.values() if v is not None]
    if not stars:
        return None
    return round(sum(stars) / len(stars) * 20)  # 1-5 stars -> 0-100


def _build_flags(category_scores: dict, rating: Optional[str], consensus: Optional[dict]) -> list:
    """Short, plain-language flags derived purely from the disclosed
    thresholds above -- no LLM. Each is a factual restatement of a
    category already scored, not a new judgment."""
    flags = []
    if category_scores["financial_health"] is not None:
        flags.append("Strong Financials" if category_scores["financial_health"] >= 4 else "Weak Financials" if category_scores["financial_health"] <= 2 else None)
    if category_scores["valuation"] is not None:
        flags.append("Attractively Valued" if category_scores["valuation"] >= 4 else "Rich Valuation" if category_scores["valuation"] <= 2 else None)
    if category_scores["momentum"] is not None:
        flags.append("Positive Momentum" if category_scores["momentum"] >= 4 else "Negative Momentum" if category_scores["momentum"] <= 2 else None)
    if category_scores["risk"] is not None:
        flags.append("Elevated Risk" if category_scores["risk"] <= 2 else None)
    if category_scores["growth"] is not None:
        flags.append("Strong Growth" if category_scores["growth"] >= 4 else None)
    if consensus and not consensus.get("low_sample_size") and consensus["score"] >= 75:
        flags.append("High Institutional Agreement")
    return [f for f in flags if f]


def build_stock_insights(ticker: str) -> dict:
    """Raises TickerNotFoundError (via MarketDataTool, run internally
    by ValuationTool) for a bad/delisted ticker -- same convention as
    every other ticker-keyed endpoint in main.py."""
    context = ResearchContext(ticker=ticker, question=f"Should I invest in {ticker}?")
    ValuationTool().run(context)

    valuation_results = context.valuation_results or {}
    currency = (context.company_info or {}).get("currency") or "USD"
    recommendation = derive_recommendation(valuation_results, None, None, currency=currency)

    alpha_factors = valuation_results.get("alpha_factors") or {}
    financial = alpha_factors.get("financial") or {}
    quality = alpha_factors.get("quality") or {}
    market = alpha_factors.get("market") or {}
    risk = alpha_factors.get("risk") or {}

    category_scores = {
        "valuation": _valuation_star(recommendation),
        "financial_health": _financial_health_star(quality),
        "growth": _growth_star(financial),
        "profitability": _profitability_star(quality, financial),
        "momentum": _momentum_star(market),
        "risk": _risk_star(risk),
    }

    rating = recommendation.get("rating")
    institutional_ratings = fetch_institutional_ratings(ticker)
    consensus = build_consensus_report(rating, institutional_ratings) if rating in ("Buy", "Hold", "Sell") else None

    return {
        "rating": rating,
        "overall_score": _overall_score(category_scores),
        "category_scores": category_scores,
        "fair_value_estimate": round(valuation_results["intrinsic_value"], 2) if valuation_results.get("intrinsic_value") else None,
        "current_price": valuation_results.get("current_price"),
        "upside_percent": valuation_results.get("upside_percent"),
        "consensus": consensus,
        "flags": _build_flags(category_scores, rating, consensus),
    }
