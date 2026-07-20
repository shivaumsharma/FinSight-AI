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

import math
from typing import Any, Dict

from app.core.research_context import ResearchContext


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)

# Recommendation thresholds, applied to the composite score below (a
# blend of the DCF and relative-valuation component scores, both on
# the same +/-percent scale) -- same numeric thresholds as before, now
# applied to the blended score rather than DCF's upside_percent alone.
BUY_THRESHOLD = 15.0
SELL_THRESHOLD = -15.0

# Composite weighting: DCF remains the primary signal (more granular,
# company-specific), but relative valuation now has real, substantial
# pull on the headline rating -- not just a confidence-flag footnote.
# When relative valuation is unavailable, DCF_WEIGHT is renormalized
# to 1.0 (see _composite_score) rather than silently discarding 40%
# of the score.
DCF_WEIGHT = 0.6
RELATIVE_WEIGHT = 0.4

# Component scores are clipped to this range before blending. Without
# a cap, an outlier DCF upside (NVDA has hit +400%+ in testing) would
# swamp the relative-valuation component regardless of DCF_WEIGHT --
# capping keeps both components on a comparable, bounded scale so the
# stated weights actually reflect how much each one influences the
# blend.
SCORE_CAP = 100.0


def _fallback_recommendation(
    relative_valuation: Dict[str, Any],
    sentiment_summary: Dict[str, Any],
    news_sentiment_summary: Dict[str, Any],
):
    """
    Used only when DCF is unavailable (see derive_recommendation
    below). Some companies -- confirmed in testing: PLTR, CVNA --
    have working-capital consumption that structurally exceeds their
    profitability, which makes FCFF-DCF produce a negative intrinsic
    value rather than a real signal; ValuationPipeline detects this
    and skips DCF entirely instead of reporting a nonsensical Sell.

    Rather than have "DCF unavailable" default to Sell (or anything
    else) through some other fallback path, the recommendation here
    is explicitly derived from the two signals that ARE available:
    relative valuation (this company's own current multiple vs. its
    3-5yr trading history) and sentiment (management commentary +
    recent news tone). Returns None if there isn't enough signal to
    derive anything (no relative valuation data either), in which
    case the caller reports "Insufficient Data" honestly rather than
    guessing.
    """
    if not relative_valuation:
        return None

    signal = relative_valuation.get("signal")

    sentiment_label = (sentiment_summary or {}).get("Overall Sentiment")
    news_label = (news_sentiment_summary or {}).get("Overall Sentiment")
    bullish_sentiment = "Positive" in (sentiment_label, news_label)
    bearish_sentiment = "Negative" in (sentiment_label, news_label)

    if signal == "cheap" and not bearish_sentiment:
        rating = "Buy"
    elif signal == "expensive" and not bullish_sentiment:
        rating = "Sell"
    else:
        # "in-line", or the two available signals point in opposite
        # directions -- don't force a Buy/Sell when they disagree.
        rating = "Hold"

    sentiment_bits = []
    if sentiment_label and sentiment_label != "Unavailable":
        sentiment_bits.append(f"management sentiment {sentiment_label.lower()}")
    if news_label and news_label != "Unavailable":
        sentiment_bits.append(f"news sentiment {news_label.lower()}")
    sentiment_desc = " and ".join(sentiment_bits) if sentiment_bits else "no sentiment signal available"

    return {
        "rating": rating,
        "basis": (
            f"DCF was unavailable ({(relative_valuation or {}).get('method', 'see valuation analysis')}), "
            f"so this recommendation is derived instead from relative valuation "
            f"({signal}, current EV/EBITDA {relative_valuation['current_ev_ebitda']:.1f}x vs. "
            f"{relative_valuation['historical_avg_ev_ebitda']:.1f}x own "
            f"{relative_valuation['years_used']}-year average) and {sentiment_desc}."
        ),
    }


def compute_signal_agreement(rating: str, relative_valuation: Dict[str, Any]):
    """
    Public: whether a Buy/Sell direction agrees with the relative
    valuation signal (Buy=cheap, Sell=expensive). "agree" / "disagree"
    / None (Hold, Insufficient Data, or no relative valuation available
    -- there's no direction to compare against). Single source of
    truth for this, used by derive_recommendation's disagreement
    downgrade below, by the forward-tracking log (prediction_log.py),
    and by the Phase 1/2 evaluation scripts, so all of them apply the
    exact same definition of "agreement."
    """
    if rating not in ("Buy", "Sell") or not relative_valuation:
        return None

    signal = relative_valuation.get("signal")
    if rating == "Sell":
        return "agree" if signal == "expensive" else "disagree"
    return "agree" if signal == "cheap" else "disagree"


def _rating_from_score(score: float) -> str:
    if score >= BUY_THRESHOLD:
        return "Buy"
    if score <= SELL_THRESHOLD:
        return "Sell"
    return "Hold"


def _dcf_score(upside_percent: float) -> float:
    return max(-SCORE_CAP, min(SCORE_CAP, upside_percent))


def _relative_score(relative_valuation: Dict[str, Any]):
    """
    Relative valuation's vs_history_pct is expensive-positive /
    cheap-negative (see relative_valuation.py) -- the opposite sign
    convention from "bullish is positive," which the DCF score and
    BUY/SELL_THRESHOLD both use. Negating it here puts both component
    scores on the same bullish-positive scale before blending.
    """
    if not relative_valuation:
        return None
    vs_history_pct = relative_valuation.get("vs_history_pct")
    if vs_history_pct is None or _is_nan(vs_history_pct):
        return None
    return max(-SCORE_CAP, min(SCORE_CAP, -vs_history_pct))


def _composite_score(dcf_score: float, relative_score):
    if relative_score is None:
        return dcf_score
    return DCF_WEIGHT * dcf_score + RELATIVE_WEIGHT * relative_score


def derive_recommendation(
    valuation_results: Dict[str, Any],
    sentiment_summary: Dict[str, Any] = None,
    news_sentiment_summary: Dict[str, Any] = None,
) -> Dict[str, str]:
    """
    Public (not module-private) since institutional_consensus_tool.py
    also calls this directly -- it independently re-derives FinSight's
    own rating from valuation_results rather than reading it back from
    an already-built report, so the consensus tool has no code path
    that could influence what this returns.

    sentiment_summary/news_sentiment_summary are optional (default
    None) so existing callers that don't have sentiment on hand keep
    working -- they just fall through to "Insufficient Data" instead
    of the relative-valuation+sentiment fallback when DCF is unavailable.

    When DCF IS available, the rating comes from a weighted composite
    of the DCF score and the relative valuation score (see
    DCF_WEIGHT/RELATIVE_WEIGHT above), not from DCF alone -- relative
    valuation now has real pull on the headline call, not just a
    confidence-flag footnote.

    On top of that blend, a hard guardrail: if DCF's OWN directional
    call (independent of the blend) actively disagrees with the
    relative valuation signal, the rating is capped at Hold regardless
    of what the composite score alone would say. Blending alone
    doesn't guarantee this -- a strongly bullish DCF can still clear
    the Buy threshold after a 40%-weighted bearish relative score is
    mixed in -- so this is enforced explicitly rather than assumed.
    """
    valuation_results = valuation_results or {}
    upside = valuation_results.get("upside_percent")

    # NaN is treated exactly like "missing" here, not like a real
    # number. ValuationPipeline now catches the one confirmed source
    # of a NaN upside (WACC going NaN when interest_expense is
    # entirely unreported -- see valuation_pipeline.py) before it ever
    # reaches this far, but this check stays as defense-in-depth: a
    # NaN silently surviving here would otherwise reach _dcf_score()'s
    # max(-100, min(100, nan)), where Python's NaN-vs-`<` comparison
    # quirk resolves that to +100 -- a maximally confident Buy for a
    # value that was never actually computed.
    if _is_nan(upside):
        upside = None

    if upside is None:
        fallback = _fallback_recommendation(
            valuation_results.get("relative_valuation"),
            sentiment_summary,
            news_sentiment_summary,
        )
        if fallback:
            return fallback

        return {
            "rating": "Insufficient Data",
            "basis": valuation_results.get("dcf_unavailable_reason")
                     or "DCF valuation or current price unavailable, and no "
                        "relative valuation/sentiment signal was available "
                        "either, so no recommendation can be derived.",
        }

    relative_valuation = valuation_results.get("relative_valuation")
    dcf_score = _dcf_score(upside)
    relative_score = _relative_score(relative_valuation)
    composite_score = _composite_score(dcf_score, relative_score)

    rating = _rating_from_score(composite_score)

    dcf_only_rating = _rating_from_score(dcf_score)
    disagreement = compute_signal_agreement(dcf_only_rating, relative_valuation) == "disagree"

    downgraded = False
    if disagreement and rating in ("Buy", "Sell"):
        rating = "Hold"
        downgraded = True

    if relative_score is None:
        weight_desc = "relative valuation unavailable, 100% DCF weight"
    else:
        weight_desc = f"relative valuation {relative_score:+.1f} x {RELATIVE_WEIGHT:.0%}"

    basis = (
        f"Composite score {composite_score:+.1f} (DCF {dcf_score:+.1f} x {DCF_WEIGHT:.0%} "
        f"+ {weight_desc}) -- Buy at >={BUY_THRESHOLD:.0f}, Sell at <={SELL_THRESHOLD:.0f}, "
        f"Hold in between."
    )
    if downgraded:
        basis += (
            f" Downgraded to Hold: DCF's own call ({dcf_only_rating}) disagrees with "
            f"relative valuation ({relative_valuation['signal']}) -- when the two signals "
            f"actively contradict each other, the recommendation is capped at Hold rather "
            f"than trusting either signal alone."
        )

    return {
        "rating": rating,
        "basis": basis,
        "dcf_score": round(dcf_score, 1),
        "relative_score": round(relative_score, 1) if relative_score is not None else None,
        "composite_score": round(composite_score, 1),
        "downgraded_for_disagreement": downgraded,
        # What the rating would have been from DCF alone, before the
        # composite blend and the disagreement guardrail -- exposed so
        # callers (e.g. the backtest) can measure whether downgrading
        # to Hold on disagreement actually improves accuracy versus
        # what the original DCF-only call would have scored.
        "dcf_only_rating": dcf_only_rating,
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

    DCF/relative-valuation disagreement is NOT flagged here anymore --
    derive_recommendation() now downgrades the rating to Hold outright
    when the two signals actively disagree (with its own explanation
    in the recommendation's "basis"), so by the time a Buy/Sell rating
    reaches this function, it has already survived that check. Flagging
    it again here would be dead code (a real disagreement can never
    produce a Buy/Sell rating anymore) and would duplicate what's
    already explained in "basis".
    """
    flags = []

    if rating in ("Buy", "Sell") and valuation_results.get("dcf_available") is False:
        flags.append(
            "This recommendation was derived without DCF -- FCFF-DCF did not apply "
            "to this company (see Valuation Analysis for why), so this call rests on "
            "relative valuation and sentiment alone, not the usual three-signal basis. "
            "Treat it as lower-confidence than a DCF-backed call."
        )

    sensitivity_df = valuation_results.get("sensitivity_analysis")
    if rating in ("Buy", "Sell") and current_price is not None and sensitivity_df is not None and not sensitivity_df.empty:

        if rating == "Sell":
            best_case_value = sensitivity_df.values.max()
            if best_case_value < current_price:
                gap = (best_case_value - current_price) / current_price * 100
                flags.append(
                    f"Even the most bullish WACC/growth combination tested (implied value "
                    f"${best_case_value:,.2f}) remains {gap:.1f}% below the current price "
                    f"of ${current_price:,.2f}. This Sell call does not vary across the "
                    f"tested sensitivity range -- that can mean the conclusion is robust, "
                    f"or that the model's assumptions (WACC, growth, or the FCF base) are "
                    f"miscalibrated for this company. Check the Institutional Consensus "
                    f"Score below before treating this as high-confidence."
                )
        else:
            worst_case_value = sensitivity_df.values.min()
            if worst_case_value > current_price:
                gap = (worst_case_value - current_price) / current_price * 100
                flags.append(
                    f"Even the most bearish WACC/growth combination tested (implied value "
                    f"${worst_case_value:,.2f}) remains {gap:.1f}% above the current price of "
                    f"${current_price:,.2f}. This Buy call does not vary across the tested "
                    f"sensitivity range -- that can mean the conclusion is robust, or that the "
                    f"model's assumptions are miscalibrated for this company. Check the "
                    f"Institutional Consensus Score below before treating this as "
                    f"high-confidence."
                )

    if not flags:
        return None

    return " ".join(flags)


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

    recommendation = derive_recommendation(valuation_results, sentiment_summary, context.news_sentiment_summary)

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
            "DCF Available": valuation_results.get("dcf_available", True),
            "DCF Unavailable Reason": valuation_results.get("dcf_unavailable_reason"),
            "Enterprise Value": valuation_summary.get("Enterprise Value", "Unavailable"),
            "Equity Value": valuation_summary.get("Equity Value", "Unavailable"),
            "Intrinsic Value (per share)": valuation_summary.get("Intrinsic Value", "Unavailable"),
            "Current Price": valuation_summary.get("Current Price", "Unavailable"),
            "Upside (%)": valuation_summary.get("Upside (%)", "Unavailable"),
            "WACC": valuation_summary.get("WACC", "Unavailable"),
            "Raw WACC": valuation_summary.get("Raw WACC", "Unavailable"),
            "WACC Floor Note": valuation_summary.get("WACC Floor Note"),
            "Terminal Growth Rate": valuation_summary.get("Terminal Growth", "Unavailable"),
            "sensitivity_table": valuation_results.get("sensitivity_analysis"),
            "relative_valuation": valuation_results.get("relative_valuation"),
            "monte_carlo": valuation_results.get("monte_carlo"),
            "ml_classifier": valuation_results.get("ml_classifier"),
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
            **recommendation,
            "confidence_flag": _confidence_flag(
                valuation_results,
                info.get("current_price"),
                recommendation["rating"],
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
