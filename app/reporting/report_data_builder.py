"""
report_data_builder.py

Assembles the numeric/structured report sections directly from
already-computed data on the ResearchContext -- no LLM involved, so
these sections are always accurate regardless of how the local model
behaves that run. Only the narrative sections (Executive Summary,
Business Analysis, Market & Earnings discussion, Risk Analysis,
Investment Thesis) are LLM-written -- see narrative_builder.py.

The Buy/Hold/Sell recommendation is a transparent, threshold-based
rule derived from the DCF's own upside/downside estimate, not
something the LLM declares from nothing -- the LLM only writes the
supporting narrative in the Investment Thesis section around
whatever this rule decided.
"""

from typing import Any, Dict

from app.core.research_context import ResearchContext

# Recommendation thresholds, applied to valuation_results["upside_percent"]
# (intrinsic value vs. current price, from the DCF model).
BUY_THRESHOLD = 15.0
SELL_THRESHOLD = -15.0


def derive_recommendation(valuation_results: Dict[str, Any]) -> Dict[str, str]:
    """
    Public (not module-private) since institutional_consensus_tool.py
    also calls this directly -- it independently re-derives FinSight's
    own rating from valuation_results rather than reading it back from
    an already-built report, so the consensus tool has no code path
    that could influence what this returns.
    """
    upside = (valuation_results or {}).get("upside_percent")

    if upside is None:
        return {
            "rating": "Insufficient Data",
            "basis": "DCF valuation or current price unavailable, so no "
                     "recommendation can be derived.",
        }

    if upside >= BUY_THRESHOLD:
        rating = "Buy"
    elif upside <= SELL_THRESHOLD:
        rating = "Sell"
    else:
        rating = "Hold"

    return {
        "rating": rating,
        "basis": f"DCF intrinsic value implies {upside:+.1f}% "
                 f"{'upside' if upside >= 0 else 'downside'} vs. the current "
                 f"price (Buy at >={BUY_THRESHOLD:.0f}%, Sell at "
                 f"<={SELL_THRESHOLD:.0f}%, Hold in between).",
    }


def _confidence_flag(valuation_results: Dict[str, Any], current_price, rating: str):
    """
    If the recommendation holds across the ENTIRE tested sensitivity
    grid (e.g. even the most bullish WACC/growth combination still
    implies Sell), surface that explicitly rather than presenting a
    Sell/Buy that never varies within the tested range as if it were
    exactly as confident as one that does.

    Deliberately does NOT change the recommendation, and deliberately
    does not claim this proves the conclusion is wrong -- a
    recommendation that holds across the whole tested grid can mean
    the conclusion is robust, or it can mean the assumptions (WACC,
    growth, FCF base) are miscalibrated for this company. Both
    readings are stated; the reader is pointed at the Institutional
    Consensus Score to help judge which.
    """
    if rating not in ("Buy", "Sell") or current_price is None:
        return None

    sensitivity_df = valuation_results.get("sensitivity_analysis")
    if sensitivity_df is None or sensitivity_df.empty:
        return None

    if rating == "Sell":
        best_case_value = sensitivity_df.values.max()
        if best_case_value >= current_price:
            return None
        gap = (best_case_value - current_price) / current_price * 100
        return (
            f"Even the most bullish WACC/growth combination tested (implied value "
            f"${best_case_value:,.2f}) remains {gap:.1f}% below the current price "
            f"of ${current_price:,.2f}. This Sell call does not vary across the "
            f"tested sensitivity range -- that can mean the conclusion is robust, "
            f"or that the model's assumptions (WACC, growth, or the FCF base) are "
            f"miscalibrated for this company. Check the Institutional Consensus "
            f"Score below before treating this as high-confidence."
        )

    worst_case_value = sensitivity_df.values.min()
    if worst_case_value <= current_price:
        return None
    gap = (worst_case_value - current_price) / current_price * 100
    return (
        f"Even the most bearish WACC/growth combination tested (implied value "
        f"${worst_case_value:,.2f}) remains {gap:.1f}% above the current price of "
        f"${current_price:,.2f}. This Buy call does not vary across the tested "
        f"sensitivity range -- that can mean the conclusion is robust, or that the "
        f"model's assumptions are miscalibrated for this company. Check the "
        f"Institutional Consensus Score below before treating this as "
        f"high-confidence."
    )


def _references(context: ResearchContext) -> list:
    """
    One unified, sequentially-numbered reference list -- SEC evidence
    and news articles both feed into it (matching the "[Evidence N]"
    / "[News N]" tags the narrative prompt is instructed to use), each
    independently checkable via a real URL where one exists. Only
    news_selected (the subset actually available to the LLM) appears
    here, not every article retrieved -- the full retrieved set is a
    separate transparency concern, see _news_sources() below.
    """
    references = []

    disclosure = (context.metadata or {}).get("disclosure_source")
    if disclosure:
        references.append({
            "type": "SEC Filing",
            "label": f"{disclosure.get('form', 'Filing')} "
                     f"({disclosure.get('filing_date', 'date unknown')})",
            "url": disclosure.get("source_url"),
        })

    for citation in context.citations or []:
        references.append({
            "type": "Retrieved Evidence",
            "label": f"{citation.get('id', 'Citation')} "
                     f"({citation.get('section', 'General')})",
            "url": None,
        })

    for article in context.news_selected or []:
        references.append({
            "type": "News",
            "label": f"{article['source']} ({article['date']}) -- {article['headline']}",
            "url": article["url"],
        })

    references.append({
        "type": "Market Data",
        "label": "Yahoo Finance (yfinance) -- prices and financial statements",
        "url": None,
    })

    return references


def _news_sources(context: ResearchContext) -> Dict[str, Any]:
    """
    Transparency data for the "News Sources Used in This Analysis"
    panel -- deliberately separate from _references() above. Shows
    EVERY article retrieved, not just the ones selected for the LLM
    prompt or explicitly cited in prose, so a reader can judge whether
    the selection looks reasonable (e.g. "did it miss an obvious
    story?") instead of only seeing what the model chose to reference.
    """
    selected_urls = {a["url"] for a in (context.news_selected or [])}

    return {
        "total_retrieved": len(context.news_articles or []),
        "total_selected": len(context.news_selected or []),
        "all_articles": [
            {**article, "used_in_analysis": article["url"] in selected_urls}
            for article in (context.news_articles or [])
        ],
    }


def build_report_data(context: ResearchContext) -> Dict[str, Any]:

    info = context.company_info or {}
    financial = context.financial_summary or {}
    valuation_summary = context.valuation_summary or {}
    valuation_results = context.valuation_results or {}
    sentiment_summary = context.sentiment_summary or {}
    evaluation = context.evaluation or {}

    return {
        "ticker": context.ticker,

        "company_overview": {
            "name": info.get("company_name") or context.ticker,
            "sector": info.get("sector", "Unavailable"),
            "industry": info.get("industry", "Unavailable"),
            "market_cap": info.get("market_cap"),
            "employees": info.get("employees"),
            "website": info.get("website"),
            "business_summary": info.get("business_summary"),
        },

        "financial_statement_analysis": {
            "Revenue": financial.get("Revenue", "Unavailable"),
            "EBIT": financial.get("EBIT", "Unavailable"),
            "Net Income": financial.get("Net Income", "Unavailable"),
            "Free Cash Flow": financial.get("Free Cash Flow", "Unavailable"),
        },

        "ratio_analysis": {
            "Operating Margin (%)": financial.get("Operating Margin", "Unavailable"),
            "Net Margin (%)": financial.get("Net Margin", "Unavailable"),
            "Return on Equity (%)": financial.get("ROE", "Unavailable"),
            "EPS": financial.get("EPS", "Unavailable"),
            "Debt to Equity": financial.get("Debt to Equity", "Unavailable"),
        },

        "growth_analysis": {
            "Revenue Growth (%)": financial.get("Revenue Growth (%)", "Unavailable"),
            "Revenue CAGR (%)": financial.get("Revenue CAGR (%)", "Unavailable"),
            "EBIT Growth (%)": financial.get("EBIT Growth (%)", "Unavailable"),
            "Net Income Growth (%)": financial.get("Net Income Growth (%)", "Unavailable"),
            "FCF Growth (%)": financial.get("FCF Growth (%)", "Unavailable"),
            "Revenue Trend": financial.get("Revenue Trend", "Unknown"),
            "Net Income Trend": financial.get("Net Income Trend", "Unknown"),
        },

        "valuation_analysis": {
            "Enterprise Value": valuation_summary.get("Enterprise Value", "Unavailable"),
            "Equity Value": valuation_summary.get("Equity Value", "Unavailable"),
            "Intrinsic Value (per share)": valuation_summary.get("Intrinsic Value", "Unavailable"),
            "Current Price": valuation_summary.get("Current Price", "Unavailable"),
            "Upside (%)": valuation_summary.get("Upside (%)", "Unavailable"),
            "WACC": valuation_summary.get("WACC", "Unavailable"),
            "Terminal Growth Rate": valuation_summary.get("Terminal Growth", "Unavailable"),
            "sensitivity_table": valuation_results.get("sensitivity_analysis"),
        },

        "market_earnings_snapshot": {
            "current_price": info.get("current_price"),
            "market_cap": info.get("market_cap"),
            # Management Sentiment: FinBERT over SEC filing evidence.
            "sentiment_label": sentiment_summary.get("Overall Sentiment", "Unavailable"),
            "sentiment_confidence": sentiment_summary.get("Confidence", "Unavailable"),
            # Market/Media Sentiment: separate FinBERT pass over recent
            # news -- shown side by side so a reader can see whether
            # management tone and independent media tone diverge.
            "news_sentiment_label": (context.news_sentiment_summary or {}).get("Overall Sentiment", "Unavailable"),
            "news_sentiment_confidence": (context.news_sentiment_summary or {}).get("Confidence", "Unavailable"),
        },

        "recommendation": {
            **derive_recommendation(valuation_results),
            "confidence_flag": _confidence_flag(
                valuation_results,
                info.get("current_price"),
                derive_recommendation(valuation_results)["rating"],
            ),
        },

        "confidence_scores": {
            "Overall Score": evaluation.get("overall_score", "Unavailable"),
            "Grounding (%)": evaluation.get("grounding_score", "Unavailable"),
            "Retrieval (%)": evaluation.get("retrieval_score", "Unavailable"),
            "Citation Coverage (%)": evaluation.get("citation_score", "Unavailable"),
            "Completeness (%)": evaluation.get("completeness_score", "Unavailable"),
        },

        "references": _references(context),

        "news_sources": _news_sources(context),

        # Populated by institutional_consensus_tool.py, which runs
        # before this. None if no institutional analyst coverage
        # exists for this ticker. Evaluation/market-context only --
        # see app/reporting/consensus_score.py's module docstring.
        "institutional_consensus": context.institutional_consensus,
    }
