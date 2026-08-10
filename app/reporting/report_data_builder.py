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
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.currency import currency_symbol
from app.core.research_context import ResearchContext


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)

# Recommendation thresholds, applied to the composite score below (a
# blend of the DCF and relative-valuation component scores, both on
# the same +/-percent scale).
#
# Both this band width and DCF_WEIGHT/RELATIVE_WEIGHT below were
# retuned from (15.0, 0.6/0.4) after backtesting (scripts/
# phase2_backtest.py + scripts/tune_recommendation_config.py +
# scripts/validate_tuning_stability.py) across TWO independent,
# non-overlapping 12-month windows: a tighter band and higher DCF
# weight beat the old config in both, not just one -- a config
# selected using only the first window also scored at the top when
# tested cold against the second window's data it never saw. The
# values below are a deliberately moderate pick, not the literal best
# scorer in either window (which was a 5-point band at 100% DCF
# weight, i.e. zeroing relative valuation's influence entirely) --
# 80/20 captured nearly all of the measured benefit in both windows
# while keeping the relative-valuation cross-check meaningfully alive,
# rather than fully retiring a feature backed by only two backtest
# windows' worth of evidence.
#
# Re-validated, not just carried over unexamined, after _dcf_score/
# _relative_score switched from raw-percentage clipping to percentile-
# rank normalization (see the "Percentile-based normalization" comment
# below): re-ran tune_recommendation_config.py + validate_tuning_stability.py
# against both windows' backtest data recomputed under the new scoring
# (scripts/backtest_results_curated_asof{12,24}mo_*.json). Both windows'
# grid-search "winners" were flagged as narrow spikes (few nearby configs
# scored similarly) AND scored WORSE than these production values on
# held-out data in the stability check (window 1: winner 58.9% vs. prod
# 61.1%; window 2: winner 41.6% vs. prod 42.5%) -- i.e. the values below
# already generalize better than anything the grid search "found," so
# they were kept unchanged rather than chased into an overfit config.
BUY_THRESHOLD = 7.5
SELL_THRESHOLD = -7.5

# Composite weighting -- see the backtest note above. DCF gets most of
# the weight (both windows showed it as the stronger standalone
# signal), with relative valuation kept at a real, non-trivial 20%
# rather than reduced to a confidence-flag footnote. When relative
# valuation is unavailable, DCF_WEIGHT is renormalized to 1.0 (see
# _composite_score) rather than silently discarding 20% of the score.
DCF_WEIGHT = 0.8
RELATIVE_WEIGHT = 0.2

# Component scores are on this range after normalization (see
# _percentile_rank_score below) -- both are now percentile ranks
# against real historical experience, rescaled from [0, 100] to
# [-100, +100] so 0 means "a perfectly median reading" and the sign
# still means bullish (+) / bearish (-), matching the scale the raw,
# pre-normalization scores used to occupy.
SCORE_CAP = 100.0

# --- Percentile-based normalization ------------------------------------
#
# DCF upside_percent and relative-valuation's vs_history_pct are both
# percentages, but NOT percentages of the same thing, and their real-
# world spreads are wildly different -- measured directly from
# scripts/backtest_results_*.json (scripts/phase2_backtest.py's saved
# output across the 79-ticker curated universe and the 1,002-ticker
# broad universe, both backtest windows: 1,341 DCF-available and 1,331
# relative-valuation-available observations). Blending both as raw
# percentages at fixed 80/20 weights (the previous approach, with each
# side independently clipped to +/-100 -- see the old SCORE_CAP
# comment, now below) was blending two different units as if they were
# the same one: a +13% DCF gap and a +18% relative-valuation discount
# do not represent comparably "extreme" readings, but raw-percentage
# blending (each side independently clipped to +/-100) scored them as
# if they did, and on a real MSFT run this was the actual reason
# relative valuation (+18.3, not really that extreme a reading) pulled
# a Sell-zone DCF call (-13.4) back into Hold by a single point.
#
# _dcf_score/_relative_score below instead rank the raw value against
# these reference distributions and rescale that 0-100 percentile to
# the same -100..+100 scale BUY_THRESHOLD/SELL_THRESHOLD/DCF_WEIGHT/
# RELATIVE_WEIGHT already used -- so "how extreme is this reading"
# is now measured in real-world rarity, not raw percent, while every
# other constant in this module keeps its existing meaning.
#
# Breakpoints are the value at percentiles 0, 5, 10, ..., 100 of the
# reference distribution; _percentile_rank_score() linearly
# interpolates between them (and clips outside the observed range,
# same as the old hard cap did -- see its docstring).
#
# DCF's table is built from the RAW, uncapped upside_percent (winsorized
# to [-300%, 1000%] -- a handful of near-zero-price tickers produced
# pathological outliers, one DCF implying +201,252% upside, that would
# otherwise single-handedly stretch the whole top of the scale and
# compress every real observation into a sliver near p99). The
# relative-valuation table is built from the already sign-flipped,
# already +/-100-capped value _relative_score() used to return
# directly -- acceptable because only 8.3% of observations actually hit
# that cap (vs. 30.4% for DCF), so this loses far less resolution than
# building DCF's table the same way would have.
#
# Reproduce: rerun scripts/phase2_backtest.py against both universes
# and windows (see EVALUATION.md section 1), collect upside_pct and
# relative_score from the saved backtest_results_*.json files, and
# recompute np.percentile(values, range(0, 101, 5)) on each (winsorizing
# upside_pct to [-300, 1000] first) -- each table below is that result,
# the value at percentiles 0, 5, 10, ..., 100 in order.
_DCF_UPSIDE_PCT_PERCENTILES = [
    -244.7, -111.4, -94.1, -83.4, -74.9, -66.6, -57.3, -46.8, -37.3, -25.5,
    -13.5, -0.1, 15.8, 31.9, 54.1, 80.7, 116.4, 167.5, 241.9, 428.6, 1000.0,
]

_RELATIVE_SCORE_PERCENTILES = [
    -100.0, -100.0, -88.4, -68.7, -56.3, -45.9, -36.8, -29.4, -24.6, -19.9,
    -15.8, -10.9, -6.8, -2.8, 2.3, 7.4, 13.3, 21.0, 30.1, 44.5, 99.2,
]


def _percentile_rank_score(raw_value: float, reference_breakpoints: List[float]) -> float:
    """
    See the "Percentile-based normalization" comment above for why
    this exists. `reference_breakpoints` is assumed evenly spaced from
    p0 to p100 (however many points it has -- the percentile axis is
    derived from its length, not hardcoded to the specific 21-point
    (0, 5, 10, ..., 100) tables above, so this stays correct if either
    table is ever rebuilt at a different resolution, and so it's
    independently testable against a small hand-built table). np.interp
    clips values outside the reference range to the nearest end
    (percentile 0 or 100) rather than extrapolating -- the same "cap
    the extremes" behavior the old flat +/-100 clip had, just against a
    real, measured distribution instead of an arbitrary round number.
    """
    percentile_axis = np.linspace(0.0, 100.0, len(reference_breakpoints))
    percentile = float(np.interp(raw_value, reference_breakpoints, percentile_axis))
    return (percentile - 50.0) * 2.0


def _fallback_recommendation(
    relative_valuation: Dict[str, Any],
    sentiment_summary: Dict[str, Any],
    news_sentiment_summary: Dict[str, Any],
    dcf_unavailable_reason: str = None,
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
            # BUG FIXED HERE: this used to read
            # relative_valuation.get("method") -- relative_valuation.py's
            # OWN methodology description ("Own trading history (no peer/
            # sector multiple data source is wired into this pipeline)"),
            # which has nothing to do with why DCF was unavailable. Found
            # by directly running this on KHC/HAS/CVNA: the basis text
            # read "DCF was unavailable (Own trading history (no peer/
            # sector multiple data source is wired into this pipeline)),"
            # which explains nothing -- and affected every DCF-unavailable
            # ticker (banks, negative-FCFF names, ...), not just the
            # leverage-instability guard below. dcf_unavailable_reason is
            # ValuationPipeline's actual, specific, per-company reason.
            f"DCF was unavailable ({dcf_unavailable_reason or 'see valuation analysis'}), "
            f"so this recommendation is derived instead from relative valuation "
            f"({signal}, current {relative_valuation['metric']} {relative_valuation['current_ev_ebitda']:.1f}x vs. "
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
    """Percentile rank of `upside_percent` against real historical DCF
    calls -- see the "Percentile-based normalization" comment above."""
    return _percentile_rank_score(upside_percent, _DCF_UPSIDE_PCT_PERCENTILES)


def _relative_score(relative_valuation: Dict[str, Any]) -> Optional[float]:
    """
    Relative valuation's vs_history_pct is expensive-positive /
    cheap-negative (see relative_valuation.py) -- the opposite sign
    convention from "bullish is positive," which the DCF score and
    BUY/SELL_THRESHOLD both use. Negating it here puts both component
    scores on the same bullish-positive scale before the percentile
    rank below (see the "Percentile-based normalization" comment
    above for why this is a percentile rank rather than the raw
    negated percentage).
    """
    if not relative_valuation:
        return None
    vs_history_pct = relative_valuation.get("vs_history_pct")
    if vs_history_pct is None or _is_nan(vs_history_pct):
        return None
    return _percentile_rank_score(-vs_history_pct, _RELATIVE_SCORE_PERCENTILES)


def _composite_score(dcf_score: float, relative_score):
    if relative_score is None:
        return dcf_score
    return DCF_WEIGHT * dcf_score + RELATIVE_WEIGHT * relative_score


def _monte_carlo_ci_straddles_price(valuation_results: Dict[str, Any]) -> bool:
    """
    True when the Monte Carlo simulation's 90% confidence interval for
    intrinsic value (app/valuation/monte_carlo_dcf.py, ci_lower/
    ci_upper -- the 5th/95th percentile across growth/WACC/terminal-
    growth uncertainty) straddles the current price -- i.e. the
    simulation itself has no real directional conviction, even if the
    DCF point estimate alone looks like a confident Buy/Sell.

    This started as a hard cap (a straddling CI forced the rating to
    Hold), matching the reasonable-sounding argument that a wide
    distribution shouldn't be presented as a confident directional
    call. Backtested (scripts/phase2_backtest.py, both curated-universe
    windows) before shipping, the same way every other threshold in
    this file was -- and it measurably hurt accuracy: on the tickers it
    actually flipped to Hold, the ORIGINAL composite call would have
    been correct 8/15 (window 1) and 8/11 (window 2), versus 3/15 and
    2/11 for the forced Hold. Both backtest windows are strongly
    bullish periods (62-72% of tickers moved >+5%), where "Hold" loses
    almost by construction -- this is the exact same failure mode
    derive_recommendation()'s docstring already documents for the old
    disagreement-forced-Hold guardrail, which was removed for the same
    measured reason. So this is now detection-only, surfaced as a
    confidence flag (see _confidence_flag) rather than overriding the
    rating -- epistemic honesty about the uncertainty without discarding
    a call that backtests better left alone.
    """
    monte_carlo = valuation_results.get("monte_carlo")
    current_price = valuation_results.get("current_price")
    if not monte_carlo or current_price is None:
        return False

    ci_lower = monte_carlo.get("ci_lower")
    ci_upper = monte_carlo.get("ci_upper")
    if ci_lower is None or ci_upper is None:
        return False

    return ci_lower <= current_price <= ci_upper


def derive_recommendation(
    valuation_results: Dict[str, Any],
    sentiment_summary: Dict[str, Any] = None,
    news_sentiment_summary: Dict[str, Any] = None,
    currency: str = "USD",
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

    Previously, a disagreement between DCF's own directional call and
    the relative valuation signal forced the rating to Hold outright,
    regardless of the composite score. Removed after Phase 2
    backtesting (scripts/phase2_backtest.py, ~80 tickers, 12-month
    forward window): on the 13 tickers where this fired, the forced
    Hold scored 15.4% accuracy versus 53.8% for trusting DCF's own
    call -- the guardrail was making the recommendation worse, not
    safer. The disagreement is now surfaced as a confidence flag (see
    _confidence_flag) instead of overriding the rating.
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
            valuation_results.get("dcf_unavailable_reason"),
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

    # Detection-only -- see _monte_carlo_ci_straddles_price's docstring
    # for why this no longer overrides `rating` (it used to, until
    # backtesting showed the override itself hurt accuracy the same
    # way the old disagreement-forced-Hold guardrail did). Surfaced as
    # a confidence flag instead (see _confidence_flag), same pattern as
    # `disagreement` above.
    mc_wide_ci = _monte_carlo_ci_straddles_price(valuation_results)

    if relative_score is None:
        weight_desc = "relative valuation unavailable, 100% DCF weight"
    else:
        weight_desc = f"relative valuation {relative_score:+.1f} x {RELATIVE_WEIGHT:.0%}"

    basis = (
        f"Composite score {composite_score:+.1f} (DCF {dcf_score:+.1f} x {DCF_WEIGHT:.0%} "
        # BUY_THRESHOLD/SELL_THRESHOLD are 7.5, not a whole number --
        # ":.0f" silently rounds 7.5/-7.5 to Python's round-half-to-even
        # "8"/"-8" (found by reading a real generated report end to
        # end: the LLM's Investment Thesis then repeated the wrong
        # "-8" back as if it were the real Sell threshold, sourced
        # straight from this string, not a model hallucination).
        f"+ {weight_desc}) -- Buy at >={BUY_THRESHOLD:g}, Sell at <={SELL_THRESHOLD:g}, "
        f"Hold in between."
    )
    if disagreement:
        basis += (
            f" Note: DCF's own directional call ({dcf_only_rating}) disagrees with "
            f"relative valuation ({relative_valuation['signal']}) -- the composite score "
            f"above already reflects both signals blended, not an override."
        )
    if mc_wide_ci and rating in ("Buy", "Sell"):
        monte_carlo = valuation_results["monte_carlo"]
        symbol = currency_symbol(currency)
        basis += (
            f" Note: the Monte Carlo simulation's 90% confidence interval for "
            f"intrinsic value ({symbol}{monte_carlo['ci_lower']:,.2f} - {symbol}{monte_carlo['ci_upper']:,.2f}) "
            f"straddles the current price ({symbol}{valuation_results['current_price']:,.2f}) -- "
            f"the simulation itself doesn't confidently say over- or under-valued once "
            f"realistic growth/WACC/terminal-growth uncertainty is accounted for. Worth "
            f"weighing alongside the DCF point estimate above, not a reason to discount "
            f"this rating outright (see Monte Carlo Simulation for the full distribution)."
        )

    return {
        "rating": rating,
        "basis": basis,
        "dcf_score": round(dcf_score, 1),
        "relative_score": round(relative_score, 1) if relative_score is not None else None,
        "composite_score": round(composite_score, 1),
        # Whether DCF's own directional call (ignoring the blend)
        # disagrees with relative valuation -- informational only.
        # This used to force the rating to Hold; removed after
        # backtesting showed that scored worse (15.4%) than trusting
        # DCF's own call (53.8%) on exactly these cases. Surfaced as a
        # confidence flag instead (see _confidence_flag).
        "signal_disagreement": disagreement,
        # What the rating would have been from DCF alone, before the
        # composite blend -- exposed so callers (e.g. the backtest)
        # can compare the blended composite against a DCF-only call.
        "dcf_only_rating": dcf_only_rating,
        # Whether the Monte Carlo 90% CI straddles the current price --
        # see _monte_carlo_ci_straddles_price's docstring for why this
        # is informational only (surfaced via _confidence_flag), not an
        # override of `rating`.
        "mc_wide_ci": mc_wide_ci,
    }


def _confidence_flag(
    valuation_results: Dict[str, Any], current_price, rating: str,
    disagreement: bool = False, mc_wide_ci: bool = False, currency: str = "USD",
):
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

    disagreement (DCF's own directional call vs. relative valuation)
    and mc_wide_ci (Monte Carlo 90% CI straddles price) are both
    flagged here rather than changing the rating -- derive_recommendation()
    used to force Hold on disagreement, and this file briefly forced
    Hold on mc_wide_ci too, but backtesting showed both overrides
    scored worse than trusting the blended composite score (see
    _monte_carlo_ci_straddles_price's docstring for the mc_wide_ci
    numbers). Both are now informational only.
    """
    flags = []

    if disagreement and rating in ("Buy", "Sell"):
        flags.append(
            "DCF's own directional call disagrees with the relative valuation "
            "signal on this one -- the rating above already reflects both blended "
            "together (see basis), not just DCF alone. Worth weighing alongside "
            "the Institutional Consensus Score below."
        )

    if mc_wide_ci and rating in ("Buy", "Sell"):
        flags.append(
            "The Monte Carlo simulation's 90% confidence interval for intrinsic "
            "value straddles the current price -- once realistic growth/WACC/"
            "terminal-growth uncertainty is accounted for, the simulation itself "
            "doesn't confidently say over- or under-valued (see Monte Carlo "
            "Simulation below for the full distribution). Worth weighing alongside "
            "the DCF point estimate the rating above is based on."
        )

    if rating in ("Buy", "Sell") and valuation_results.get("dcf_available") is False:
        flags.append(
            "This recommendation was derived without DCF -- FCFF-DCF did not apply "
            "to this company (see Valuation Analysis for why), so this call rests on "
            "relative valuation and sentiment alone, not the usual three-signal basis. "
            "Treat it as lower-confidence than a DCF-backed call."
        )

    sensitivity_df = valuation_results.get("sensitivity_analysis")
    if rating in ("Buy", "Sell") and current_price is not None and sensitivity_df is not None and not sensitivity_df.empty:

        symbol = currency_symbol(currency)
        if rating == "Sell":
            best_case_value = sensitivity_df.values.max()
            if best_case_value < current_price:
                gap = (best_case_value - current_price) / current_price * 100
                flags.append(
                    f"Even the most bullish WACC/growth combination tested (implied value "
                    f"{symbol}{best_case_value:,.2f}) remains {gap:.1f}% below the current price "
                    f"of {symbol}{current_price:,.2f}. This Sell call does not vary across the "
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
                    f"{symbol}{worst_case_value:,.2f}) remains {gap:.1f}% above the current price of "
                    f"{symbol}{current_price:,.2f}. This Buy call does not vary across the tested "
                    f"sensitivity range -- that can mean the conclusion is robust, or that the "
                    f"model's assumptions are miscalibrated for this company. Check the "
                    f"Institutional Consensus Score below before treating this as "
                    f"high-confidence."
                )

    if not flags:
        return None

    return " ".join(flags)


# ---------------------------------------------------------------- signal quality (display-only)
#
# Five signals already exist and are already computed on every report,
# but were never combined into one reader-facing view: confidence_scores'
# Overall Score, signal_disagreement/mc_wide_ci (already on `recommendation`,
# see derive_recommendation above), Alpha Factors' own financial/market
# categories, Management Sentiment, and real institutional analyst
# ratings. derive_signal_quality() below is the consolidation -- called
# strictly AFTER derive_recommendation() returns (see build_report_data)
# and reads only already-computed values from it and other already-built
# dicts. Same non-negotiable boundary as Alpha Factors and the ML
# classifier (see alpha_factors.py's own module docstring): display-only,
# never written back into `recommendation`/_composite_score.
#
# Thresholds below are disclosed, adjustable constants, NOT backtested
# like BUY_THRESHOLD/SELL_THRESHOLD above -- "signal quality" has no
# ground-truth outcome (a stock doesn't go up or down because the
# evidence behind a call was thin) to score a threshold choice against
# the way a directional call does.
QUALITY_HIGH_THRESHOLD = 70
QUALITY_LOW_THRESHOLD = 40
CONFIDENCE_DISAGREEMENT_PENALTY = 15
CONFIDENCE_WIDE_CI_PENALTY = 15

_ML_VERDICT_TO_RATING = {"UNDERVALUED": "Buy", "FAIRLY VALUED": "Hold", "OVERVALUED": "Sell"}
_RELATIVE_SIGNAL_TO_RATING = {"cheap": "Buy", "in-line": "Hold", "expensive": "Sell"}
_INSTITUTIONAL_RATING_TO_DIRECTION = {"BUY": "Bullish", "HOLD": "Neutral", "SELL": "Bearish"}
_SENTIMENT_LABEL_TO_DIRECTION = {"Positive": "Bullish", "Negative": "Bearish", "Neutral": "Neutral"}


def _direction_label(value: Optional[float], bullish_at: float = 0.0, bearish_at: float = 0.0) -> str:
    """Sign-based Bullish/Bearish/Neutral label for an already-computed
    percentage -- shared by signal_quality's breakdown so fundamentals/
    momentum/valuation don't each reimplement the same three-way split.
    None (missing/unavailable -- e.g. a .NS ticker with no benchmark
    history, see alpha_factors.py's _market_note) degrades to "Neutral",
    the safest default when there's nothing to judge, not a guess."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Neutral"
    if value > bullish_at:
        return "Bullish"
    if value < bearish_at:
        return "Bearish"
    return "Neutral"


def _institutional_direction(institutional_consensus: Optional[Dict[str, Any]]) -> str:
    """Majority vote among covering institutions' OWN ratings (BUY/HOLD/
    SELL) -- independent of whether they agree with FinSight's rating
    (that's consensus_score.py's separate job). Reuses the same
    institutional_ratings list already fetched for Institutional
    Consensus, no new data source. Ties resolve to whichever of
    BUY/HOLD/SELL is checked first below -- a disclosed simplification,
    not a strict-majority guarantee."""
    consensus = (institutional_consensus or {}).get("recommendation_consensus") or {}
    ratings = consensus.get("institutional_ratings") or []
    if not ratings:
        return "Neutral"
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for r in ratings:
        if r.get("rating") in counts:
            counts[r["rating"]] += 1
    top_rating = max(counts, key=counts.get)
    return _INSTITUTIONAL_RATING_TO_DIRECTION[top_rating] if counts[top_rating] else "Neutral"


def derive_signal_quality(
    recommendation: Dict[str, Any],
    valuation_results: Dict[str, Any],
    evaluation: Dict[str, Any],
    sentiment_summary: Dict[str, Any],
    institutional_consensus: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Separates *direction* (the Buy/Hold/Sell rating itself, already
    final by the time this runs) from *quality of evidence* (how much
    of the supporting machinery actually agrees, and how well-grounded
    the underlying report is) -- the two were previously conflated: a
    thinly-evidenced Buy and a thoroughly-corroborated Buy read
    identically on the report today.

    Returns {quality_label: HIGH|MEDIUM|LOW, confidence: 0-100 or None,
    evidence_coverage: 0-100 or None, model_agreement: "N/M" or "N/A",
    breakdown: {valuation, fundamentals, momentum, sentiment,
    institutional_consensus: each Bullish/Bearish/Neutral}}.
    """
    rating = recommendation.get("rating")

    overall_score = evaluation.get("overall_score")
    evidence_coverage = overall_score if isinstance(overall_score, (int, float)) else None

    confidence = evidence_coverage
    if confidence is not None:
        if recommendation.get("signal_disagreement"):
            confidence -= CONFIDENCE_DISAGREEMENT_PENALTY
        if recommendation.get("mc_wide_ci"):
            confidence -= CONFIDENCE_WIDE_CI_PENALTY
        confidence = max(0.0, min(100.0, confidence))

    if confidence is None or confidence < QUALITY_LOW_THRESHOLD:
        quality_label = "LOW"
    elif confidence >= QUALITY_HIGH_THRESHOLD:
        quality_label = "HIGH"
    else:
        quality_label = "MEDIUM"

    # model_agreement: a 3-way vote among independently-derived
    # directional calls -- DCF alone (ignoring the relative-valuation
    # blend), the relative-valuation cross-check's own signal, and the
    # ML classifier's verdict -- against the FINAL rating. Genuinely new
    # only in that these three were never compared against each other
    # before; each already exists and is already computed.
    votes = []
    if recommendation.get("dcf_only_rating"):
        votes.append(recommendation["dcf_only_rating"])

    relative_signal = (valuation_results.get("relative_valuation") or {}).get("signal")
    if relative_signal in _RELATIVE_SIGNAL_TO_RATING:
        votes.append(_RELATIVE_SIGNAL_TO_RATING[relative_signal])

    ml_verdict = (valuation_results.get("ml_classifier") or {}).get("verdict")
    if ml_verdict in _ML_VERDICT_TO_RATING:
        votes.append(_ML_VERDICT_TO_RATING[ml_verdict])

    if votes and rating in ("Buy", "Hold", "Sell"):
        model_agreement = f"{sum(1 for v in votes if v == rating)}/{len(votes)}"
    else:
        model_agreement = "N/A"

    alpha_factors = valuation_results.get("alpha_factors") or {}
    financial = alpha_factors.get("financial") or {}
    market = alpha_factors.get("market") or {}
    composite_score = recommendation.get("composite_score")

    return {
        "quality_label": quality_label,
        "confidence": round(confidence) if confidence is not None else None,
        "evidence_coverage": round(evidence_coverage) if evidence_coverage is not None else None,
        "model_agreement": model_agreement,
        "breakdown": {
            "valuation": (
                _direction_label(composite_score, bullish_at=BUY_THRESHOLD, bearish_at=SELL_THRESHOLD)
                if isinstance(composite_score, (int, float)) else "Neutral"
            ),
            "fundamentals": _direction_label(financial.get("Revenue Growth YoY (%)")),
            "momentum": _direction_label(market.get("6-Month Price Momentum (%)")),
            "sentiment": _SENTIMENT_LABEL_TO_DIRECTION.get(
                (sentiment_summary or {}).get("Overall Sentiment"), "Neutral"
            ),
            "institutional_consensus": _institutional_direction(institutional_consensus),
        },
    }


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
        filing_date = citation.get("filing_date")
        label = f"{citation.get('id', 'Citation')} ({citation.get('section', 'General')})"
        if filing_date:
            label += f" -- filed {filing_date}"
        references.append({
            "type": "Retrieved Evidence",
            "label": label,
            # CitationEngine.build() stamps every chunk from one
            # ingestion with the SAME filing's source_url (they're all
            # chunks of that one document) -- was hardcoded None here
            # even though the document-level "SEC Filing" entry two
            # lines above already had the real URL the whole time.
            "url": citation.get("source_url"),
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
    currency = info.get("currency") or "USD"

    recommendation = derive_recommendation(
        valuation_results, sentiment_summary, context.news_sentiment_summary, currency
    )

    # Strictly downstream of derive_recommendation() above -- reads its
    # output, never influences it. See derive_signal_quality's own
    # docstring/module comment for the display-only boundary.
    signal_quality = derive_signal_quality(
        recommendation, valuation_results, evaluation, sentiment_summary,
        context.institutional_consensus,
    )

    # NSE-listed tickers (see company_resolver.py's NSE index) carry a
    # ".NS" suffix and structurally have no SEC filings -- RAGTool/
    # SECEdgarClient already degrade gracefully with zero filings found
    # (empty citations, sentiment skipped), but that looks identical to
    # "we tried and found nothing" for a US ticker with genuinely thin
    # coverage. Surfaced explicitly here so a reader sees this is
    # expected market coverage, not a silent retrieval gap.
    is_non_us_listing = (context.ticker or "").upper().endswith(".NS")
    filing_evidence_note = (
        "SEC filing evidence is unavailable for this market -- FinSight's evidence "
        "retrieval currently covers SEC EDGAR (US-listed companies) only. Management "
        "Sentiment below and its supporting citations don't apply here; market data, "
        "valuation, and news sentiment are unaffected."
        if is_non_us_listing else None
    )

    return {
        "ticker": context.ticker,
        "currency": currency,
        "currency_symbol": currency_symbol(currency),
        "filing_evidence_note": filing_evidence_note,

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
            "FCF Trend": financial.get("FCF Trend", "Unknown"),
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
            "alpha_factors": valuation_results.get("alpha_factors"),
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
            # None if yfinance doesn't report one -- see
            # MarketDataLoader.get_next_earnings_date(). Consumed by
            # narrative_builder.py to flag an imminent earnings date
            # explicitly rather than leaving it to be noticed (or not)
            # among the numbered news items.
            "next_earnings_date": info.get("next_earnings_date"),
        },

        "recommendation": {
            **recommendation,
            "confidence_flag": _confidence_flag(
                valuation_results,
                info.get("current_price"),
                recommendation["rating"],
                recommendation.get("signal_disagreement", False),
                recommendation.get("mc_wide_ci", False),
                currency,
            ),
        },

        "confidence_scores": {
            "Overall Score": evaluation.get("overall_score", "Unavailable"),
            "Grounding (%)": evaluation.get("grounding_score", "Unavailable"),
            "Retrieval (%)": evaluation.get("retrieval_score", "Unavailable"),
            "Citation Coverage (%)": evaluation.get("citation_score", "Unavailable"),
            "Completeness (%)": evaluation.get("completeness_score", "Unavailable"),
        },

        # Display-only consolidation of signals already computed above --
        # see derive_signal_quality's own docstring for the display-only
        # boundary and exactly which already-computed values feed it.
        "signal_quality": signal_quality,

        "references": _references(context),

        "news_sources": _news_sources(context),

        # Populated by institutional_consensus_tool.py, which runs
        # before this. None if no institutional analyst coverage
        # exists for this ticker. Evaluation/market-context only --
        # see app/reporting/consensus_score.py's module docstring.
        "institutional_consensus": context.institutional_consensus,
    }
