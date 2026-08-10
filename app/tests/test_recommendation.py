"""
Unit tests for the recommendation formula (app/reporting/report_data_builder.py)
-- the single most consequential piece of deterministic logic in the
project, since it decides Buy/Hold/Sell before the LLM ever writes a
word. Includes a regression test for the NaN-to-+100 bug: Python's
`nan < x` is always False, so an un-guarded `max(-100, min(100, nan))`
silently resolves to +100 ("maximally confident Buy") instead of
propagating as unknown -- this must never happen again.

Also covers the percentile-rank normalization _dcf_score/_relative_score
use (see report_data_builder.py's "Percentile-based normalization"
comment): DCF upside_percent and relative valuation's vs_history_pct are
both percentages but not comparable ones (measured std/spread differ by
roughly 1.7x, and DCF hits the old flat +/-100 cap on 30.4% of real
observations vs. 8.3% for relative valuation), so both are now ranked
against real historical experience (scripts/backtest_results_*.json)
before blending, instead of independently clipping each to a flat
+/-100 and blending the raw percentages as if they meant the same thing.
"""

import math

import pytest

from app.reporting.report_data_builder import (
    BUY_THRESHOLD, SELL_THRESHOLD, DCF_WEIGHT, RELATIVE_WEIGHT, SCORE_CAP,
    QUALITY_HIGH_THRESHOLD, QUALITY_LOW_THRESHOLD,
    _DCF_UPSIDE_PCT_PERCENTILES,
    _dcf_score, _relative_score, _composite_score, _percentile_rank_score,
    _rating_from_score, compute_signal_agreement, derive_recommendation,
    derive_signal_quality,
)


# ---------------------------------------------------------- _percentile_rank_score

def test_percentile_rank_score_interpolates_between_breakpoints():
    # A small hand-built table (p0, p25, p50, p75, p100) makes the
    # interpolation arithmetic independently checkable, rather than
    # trusting the real (backtest-derived, 21-point) tables to happen
    # to prove this.
    table = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _percentile_rank_score(20.0, table) == pytest.approx(0.0)     # exactly p50
    assert _percentile_rank_score(0.0, table) == pytest.approx(-100.0)   # exactly p0
    assert _percentile_rank_score(40.0, table) == pytest.approx(100.0)   # exactly p100
    # Halfway between p25 (10.0) and p50 (20.0) -> percentile 37.5 -> (37.5-50)*2
    assert _percentile_rank_score(15.0, table) == pytest.approx(-25.0)


def test_percentile_rank_score_clips_beyond_the_table_range():
    table = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _percentile_rank_score(-1000.0, table) == pytest.approx(-100.0)
    assert _percentile_rank_score(1000.0, table) == pytest.approx(100.0)


# ---------------------------------------------------------------- _dcf_score

def test_dcf_score_is_a_percentile_rank_not_a_raw_passthrough():
    # A raw upside somewhere in real historical experience must NOT
    # come back out unchanged -- that was the old (buggy) behavior
    # this replaced. 12.5% upside sits above the reference median
    # (-13.5%), so it should score positive but nowhere near the raw
    # 12.5 a passthrough would have returned.
    score = _dcf_score(12.5)
    assert score != 12.5
    assert 0 < score < 50


def test_dcf_score_is_monotonically_increasing():
    values = [-500, -100, -13.5, 0, 12.5, 50, 200, 2000]
    scores = [_dcf_score(v) for v in values]
    assert scores == sorted(scores)


def test_dcf_score_at_the_reference_median_is_neutral():
    # _DCF_UPSIDE_PCT_PERCENTILES[10] is the p50 breakpoint (index 10 of
    # the 21-point 0,5,...,100 table) -- a value exactly at the
    # reference median must score to (50-50)*2 == 0 by construction.
    median_upside = _DCF_UPSIDE_PCT_PERCENTILES[10]
    assert _dcf_score(median_upside) == pytest.approx(0.0)


@pytest.mark.parametrize("upside,expected", [(5000, SCORE_CAP), (-500, -SCORE_CAP)])
def test_dcf_score_is_capped_at_the_reference_distributions_extremes(upside, expected):
    # Values beyond the reference table's observed range (the DCF table
    # was winsorized to [-300%, 1000%] raw upside when built -- see
    # report_data_builder.py's module-level comment) clip to the same
    # +/-100 as before, just anchored to the real extremes of historical
    # experience instead of an arbitrary +/-100%-raw-upside threshold.
    assert _dcf_score(upside) == expected


# ----------------------------------------------------------- _relative_score

def test_relative_score_flips_sign_of_vs_history_pct():
    # relative_valuation.py: positive vs_history_pct = expensive (bearish).
    # _relative_score puts everything on the same bullish-positive scale
    # before percentile-ranking -- so cheap (negative vs_history_pct)
    # must score positive and expensive (positive vs_history_pct) must
    # score negative, even though the exact magnitude is no longer a
    # straight passthrough of the input.
    assert _relative_score({"vs_history_pct": 20.0}) < 0
    assert _relative_score({"vs_history_pct": -20.0}) > 0


def test_relative_score_is_monotonically_increasing_as_valuation_gets_cheaper():
    # vs_history_pct decreasing (cheaper) must never produce a lower score.
    values = [50.0, 20.0, 0.0, -20.0, -50.0]
    scores = [_relative_score({"vs_history_pct": v}) for v in values]
    assert scores == sorted(scores)


def test_relative_score_is_none_when_unavailable():
    assert _relative_score(None) is None
    assert _relative_score({}) is None


def test_relative_score_treats_nan_as_none():
    assert _relative_score({"vs_history_pct": float("nan")}) is None


# ---------------------------------------------------------- _composite_score

def test_composite_score_blends_by_configured_weights():
    composite = _composite_score(dcf_score=10.0, relative_score=-5.0)
    assert composite == pytest.approx(DCF_WEIGHT * 10.0 + RELATIVE_WEIGHT * -5.0)


def test_composite_score_is_100pct_dcf_when_relative_unavailable():
    assert _composite_score(dcf_score=42.0, relative_score=None) == 42.0


# ---------------------------------------------------------- _rating_from_score

@pytest.mark.parametrize("score,expected", [
    (BUY_THRESHOLD, "Buy"),
    (BUY_THRESHOLD + 50, "Buy"),
    (SELL_THRESHOLD, "Sell"),
    (SELL_THRESHOLD - 50, "Sell"),
    (0.0, "Hold"),
])
def test_rating_from_score_thresholds(score, expected):
    assert _rating_from_score(score) == expected


# ---------------------------------------------------------- derive_recommendation

def _valuation(
    upside_percent, relative_valuation=None, dcf_available=True,
    monte_carlo=None, current_price=None,
):
    return {
        "upside_percent": upside_percent,
        "relative_valuation": relative_valuation,
        "dcf_available": dcf_available,
        "dcf_unavailable_reason": None if dcf_available else "test reason",
        "monte_carlo": monte_carlo,
        "current_price": current_price,
    }


def _mc(ci_lower, ci_upper):
    return {"ci_lower": ci_lower, "ci_upper": ci_upper}


def test_derive_recommendation_buy_from_strong_upside():
    result = derive_recommendation(_valuation(upside_percent=50.0))
    assert result["rating"] == "Buy"
    # dcf_score is now a percentile rank, not upside_percent itself
    # (see _dcf_score's percentile-rank tests above) -- the qualitative
    # invariant that matters here is "strong upside clears the Buy bar
    # with no relative valuation diluting it," not the exact number.
    assert result["dcf_score"] == pytest.approx(_dcf_score(50.0), abs=0.05)  # result rounds to 1 decimal
    assert result["composite_score"] == result["dcf_score"]  # no relative valuation -> 100% DCF weight


def test_derive_recommendation_sell_from_strong_downside():
    result = derive_recommendation(_valuation(upside_percent=-50.0))
    assert result["rating"] == "Sell"


def test_derive_recommendation_basis_states_the_real_threshold_not_rounded():
    """
    Regression test: BUY_THRESHOLD/SELL_THRESHOLD are 7.5/-7.5, not
    whole numbers -- an old ":.0f" format silently applied Python's
    round-half-to-even and displayed "8"/"-8" in the basis text shown
    to users AND fed into the narrative LLM prompt (caught by reading
    a real generated MSFT report end to end: the model's Investment
    Thesis repeated "-8" back as the Sell threshold, sourced straight
    from this string). Must show the real threshold value.
    """
    result = derive_recommendation(_valuation(upside_percent=50.0))
    assert f"Buy at >={BUY_THRESHOLD:g}" in result["basis"]
    assert f"Sell at <={SELL_THRESHOLD:g}" in result["basis"]
    assert "Buy at >=8" not in result["basis"]
    assert "Sell at <=-8" not in result["basis"]


def test_derive_recommendation_nan_upside_never_becomes_a_confident_buy():
    """
    Regression test: a NaN upside_percent must resolve to "Insufficient
    Data" (or the relative/sentiment fallback), NEVER to a numeric
    composite score -- let alone the +100 that an unguarded
    max(-100, min(100, nan)) would silently produce, because Python's
    `nan < x` comparisons are always False.
    """
    result = derive_recommendation(_valuation(upside_percent=float("nan")))
    assert result["rating"] != "Buy"
    assert "composite_score" not in result or result["rating"] == "Insufficient Data"


def test_derive_recommendation_insufficient_data_when_nothing_available():
    result = derive_recommendation(_valuation(upside_percent=None, dcf_available=False))
    assert result["rating"] == "Insufficient Data"


def test_derive_recommendation_falls_back_to_relative_valuation_when_dcf_unavailable():
    result = derive_recommendation(
        _valuation(
            upside_percent=None,
            relative_valuation={
                "signal": "cheap", "method": "test", "metric": "EV/EBITDA",
                "current_ev_ebitda": 8.0, "historical_avg_ev_ebitda": 12.0,
                "years_used": 4,
            },
            dcf_available=False,
        ),
        sentiment_summary={"Overall Sentiment": "Neutral"},
    )
    assert result["rating"] == "Buy"


def test_derive_recommendation_fallback_basis_states_the_real_unavailable_reason():
    """
    Regression test: _fallback_recommendation's basis text used to be
    built from relative_valuation.get("method") -- relative_valuation.py's
    own methodology description ("Own trading history (no peer/sector
    multiple data source is wired into this pipeline)"), which has
    nothing to do with why DCF was unavailable. Found by running this
    on real KHC/HAS/CVNA data: the basis text explained nothing. Must
    use the real, specific, per-company dcf_unavailable_reason instead.
    """
    result = derive_recommendation(
        _valuation(
            upside_percent=None,
            relative_valuation={
                "signal": "cheap", "method": "Own trading history (no peer/sector multiple "
                                              "data source is wired into this pipeline)",
                "metric": "EV/EBITDA", "current_ev_ebitda": 8.0, "historical_avg_ev_ebitda": 12.0,
                "years_used": 4,
            },
            dcf_available=False,
        ),
    )
    assert "test reason" in result["basis"]
    assert "Own trading history" not in result["basis"]


def test_derive_recommendation_flags_disagreement_without_forcing_hold():
    """
    A disagreement between DCF's own directional call and the relative
    valuation signal used to force a Hold outright; backtesting showed
    that scored worse than trusting the composite. It must now only be
    flagged (signal_disagreement=True), never override the rating.
    """
    result = derive_recommendation(_valuation(
        upside_percent=50.0,  # DCF alone says Buy
        relative_valuation={"vs_history_pct": 50.0, "signal": "expensive"},  # relative says expensive/Sell-ish
    ))
    assert result["signal_disagreement"] is True
    assert result["dcf_only_rating"] == "Buy"
    # The composite blend (80% DCF Buy score + 20% strongly negative
    # relative score) may or may not still clear the Buy bar -- the
    # important invariant is that disagreement does NOT force Hold.
    assert result["rating"] != "Hold" or _composite_score(50.0, -50.0) <= BUY_THRESHOLD


# ------------------------------------------ _monte_carlo_ci_straddles_price
#
# This started as a hard cap to Hold. Backtested before shipping (both
# curated-universe windows, same methodology as everything else in this
# file): on the tickers it actually flipped to Hold, the ORIGINAL
# composite call would have been correct 8/15 (window 1) and 8/11
# (window 2), versus 3/15 and 2/11 for the forced Hold -- the same
# failure mode already documented above for the old disagreement-
# forced-Hold guardrail (Hold loses almost by construction in a
# persistently bullish backtest window). So this is detection-only now,
# surfaced via _confidence_flag, never an override of `rating`.

def test_monte_carlo_ci_straddles_price_true_when_price_is_inside_the_ci():
    from app.reporting.report_data_builder import _monte_carlo_ci_straddles_price

    valuation = _valuation(
        upside_percent=50.0, monte_carlo=_mc(ci_lower=80.0, ci_upper=200.0), current_price=150.0,
    )
    assert _monte_carlo_ci_straddles_price(valuation) is True


def test_monte_carlo_ci_straddles_price_false_when_price_is_outside_the_ci():
    from app.reporting.report_data_builder import _monte_carlo_ci_straddles_price

    valuation = _valuation(
        upside_percent=50.0, monte_carlo=_mc(ci_lower=60.0, ci_upper=100.0), current_price=150.0,
    )
    assert _monte_carlo_ci_straddles_price(valuation) is False


@pytest.mark.parametrize("valuation", [
    _valuation(upside_percent=50.0, monte_carlo=None, current_price=150.0),
    _valuation(upside_percent=50.0, monte_carlo=_mc(80.0, 200.0), current_price=None),
    _valuation(upside_percent=50.0, monte_carlo={"ci_lower": None, "ci_upper": 200.0}, current_price=150.0),
])
def test_monte_carlo_ci_straddles_price_false_when_data_is_missing(valuation):
    from app.reporting.report_data_builder import _monte_carlo_ci_straddles_price

    assert _monte_carlo_ci_straddles_price(valuation) is False


def test_derive_recommendation_surfaces_wide_ci_without_overriding_the_rating():
    """
    The real motivating case (real MSFT numbers: CI $209.59-$588.27
    around a $393.35 price) -- the composite call is preserved exactly
    as the composite score determined it; mc_wide_ci is surfaced for
    the caller to turn into a confidence flag (see _confidence_flag),
    not used to change `rating` here.
    """
    result = derive_recommendation(_valuation(
        upside_percent=50.0,
        monte_carlo=_mc(ci_lower=209.59, ci_upper=588.27),
        current_price=393.35,
    ))
    assert result["rating"] == "Buy"
    assert result["mc_wide_ci"] is True


def test_derive_recommendation_mc_wide_ci_is_false_when_ci_does_not_straddle_price():
    result = derive_recommendation(_valuation(
        upside_percent=50.0,
        monte_carlo=_mc(ci_lower=200.0, ci_upper=300.0),
        current_price=150.0,  # below the entire CI -- undervalued per the simulation too
    ))
    assert result["rating"] == "Buy"
    assert result["mc_wide_ci"] is False
    assert "Monte Carlo" not in result["basis"]


# ---------------------------------------------------------- _confidence_flag

def test_confidence_flag_surfaces_wide_monte_carlo_ci_for_buy_sell_only():
    from app.reporting.report_data_builder import _confidence_flag

    flag = _confidence_flag({"dcf_available": True}, 393.35, "Buy", mc_wide_ci=True)
    assert flag is not None
    assert "Monte Carlo" in flag

    # Hold already IS the hedge -- no need to caveat it further.
    assert _confidence_flag({"dcf_available": True}, 393.35, "Hold", mc_wide_ci=True) is None


# ---------------------------------------------------------- compute_signal_agreement

def test_compute_signal_agreement_buy_matches_cheap():
    assert compute_signal_agreement("Buy", {"signal": "cheap"}) == "agree"
    assert compute_signal_agreement("Buy", {"signal": "expensive"}) == "disagree"


def test_compute_signal_agreement_sell_matches_expensive():
    assert compute_signal_agreement("Sell", {"signal": "expensive"}) == "agree"
    assert compute_signal_agreement("Sell", {"signal": "cheap"}) == "disagree"


def test_compute_signal_agreement_none_for_hold_or_missing_data():
    assert compute_signal_agreement("Hold", {"signal": "cheap"}) is None
    assert compute_signal_agreement("Buy", None) is None


# ---------------------------------------------------------- alpha_factors isolation
#
# The single most important test in this file (arguably in the whole
# Alpha Factors feature, see app/analysis/alpha_factors.py's own module
# docstring): derive_recommendation must be BYTE-FOR-BYTE identical
# regardless of whether valuation_results["alpha_factors"] is present,
# absent, or None. This is the automated proof of that non-negotiable
# boundary, not just a code-review claim -- derive_recommendation was
# independently confirmed (by reading its full body) to never reference
# the "alpha_factors" key at all, and this test would catch it the day
# that ever stops being true.

def test_derive_recommendation_ignores_alpha_factors_entirely():
    base = _valuation(
        upside_percent=22.0,
        relative_valuation={"signal": "cheap", "vs_history_pct": -15.0},
        monte_carlo=_mc(90.0, 110.0),
        current_price=100.0,
    )
    without_key = dict(base)
    with_none = {**base, "alpha_factors": None}
    with_populated = {
        **base,
        "alpha_factors": {
            "financial": {"Revenue CAGR (%)": 12.5},
            "quality": {"Altman Z-Score": 5.08, "Piotroski F-Score (0-9, proxy)": 9},
            "valuation": {"P/E vs Own History": {"signal": "expensive", "vs_history_pct": 100.0}},
            "market": {"Relative Strength vs Index (%)": 8.2},
            "risk": {"Beta": 1.35},
            "sentiment": {"Management/SEC Filing Sentiment": "Positive"},
            "macro": {"Interest Rate Sensitivity": -0.42},
        },
    }

    result_without = derive_recommendation(without_key)
    result_none = derive_recommendation(with_none)
    result_populated = derive_recommendation(with_populated)

    assert result_without == result_none == result_populated


# ---------------------------------------------------------- derive_signal_quality

def _rec(rating="Buy", composite_score=10.0, dcf_only_rating="Buy",
         signal_disagreement=False, mc_wide_ci=False):
    return {
        "rating": rating,
        "composite_score": composite_score,
        "dcf_only_rating": dcf_only_rating,
        "signal_disagreement": signal_disagreement,
        "mc_wide_ci": mc_wide_ci,
    }


def test_signal_quality_high_at_or_above_the_high_threshold():
    result = derive_signal_quality(_rec(), {}, {"overall_score": QUALITY_HIGH_THRESHOLD}, {}, None)
    assert result["quality_label"] == "HIGH"
    assert result["confidence"] == QUALITY_HIGH_THRESHOLD
    assert result["evidence_coverage"] == QUALITY_HIGH_THRESHOLD


def test_signal_quality_low_below_the_low_threshold():
    result = derive_signal_quality(_rec(), {}, {"overall_score": QUALITY_LOW_THRESHOLD - 1}, {}, None)
    assert result["quality_label"] == "LOW"


def test_signal_quality_medium_between_the_two_thresholds():
    midpoint = (QUALITY_LOW_THRESHOLD + QUALITY_HIGH_THRESHOLD) / 2
    result = derive_signal_quality(_rec(), {}, {"overall_score": midpoint}, {}, None)
    assert result["quality_label"] == "MEDIUM"


def test_signal_quality_is_low_when_evidence_coverage_is_unavailable():
    # No numeric "Overall Score" -- nothing to be confident about, so
    # this must degrade to LOW rather than crash or silently pass as
    # HIGH/MEDIUM.
    result = derive_signal_quality(_rec(), {}, {"overall_score": "Unavailable"}, {}, None)
    assert result["quality_label"] == "LOW"
    assert result["confidence"] is None
    assert result["evidence_coverage"] is None


def test_signal_quality_confidence_is_docked_for_disagreement_and_wide_ci():
    baseline = derive_signal_quality(_rec(), {}, {"overall_score": 90}, {}, None)
    docked = derive_signal_quality(
        _rec(signal_disagreement=True, mc_wide_ci=True), {}, {"overall_score": 90}, {}, None
    )
    assert docked["confidence"] < baseline["confidence"]
    assert docked["evidence_coverage"] == baseline["evidence_coverage"]  # evidence_coverage itself is undocked


def test_signal_quality_confidence_never_drops_below_zero():
    result = derive_signal_quality(
        _rec(signal_disagreement=True, mc_wide_ci=True), {}, {"overall_score": 5}, {}, None
    )
    assert result["confidence"] >= 0


def test_model_agreement_is_na_when_no_votes_are_available():
    result = derive_signal_quality(_rec(dcf_only_rating=None), {}, {}, {}, None)
    assert result["model_agreement"] == "N/A"


def test_model_agreement_is_na_for_insufficient_data_rating():
    result = derive_signal_quality(
        _rec(rating="Insufficient Data", dcf_only_rating="Buy"),
        {"relative_valuation": {"signal": "cheap"}}, {}, {}, None,
    )
    assert result["model_agreement"] == "N/A"


def test_model_agreement_counts_agreeing_votes_against_the_final_rating():
    valuation_results = {
        "relative_valuation": {"signal": "cheap"},  # -> Buy
        "ml_classifier": {"verdict": "OVERVALUED"},  # -> Sell
    }
    result = derive_signal_quality(
        _rec(rating="Buy", dcf_only_rating="Buy"), valuation_results, {}, {}, None,
    )
    # DCF-only (Buy) agrees, relative valuation (Buy) agrees, ML classifier (Sell) disagrees.
    assert result["model_agreement"] == "2/3"


def test_signal_quality_breakdown_degrades_to_neutral_when_data_is_missing():
    # No alpha_factors, no sentiment, no institutional_consensus -- must
    # never crash, and every dimension without real data reads Neutral.
    result = derive_signal_quality(_rec(dcf_only_rating=None), {}, {}, {}, None)
    breakdown = result["breakdown"]
    assert breakdown["fundamentals"] == "Neutral"
    assert breakdown["momentum"] == "Neutral"
    assert breakdown["sentiment"] == "Neutral"
    assert breakdown["institutional_consensus"] == "Neutral"


def test_signal_quality_breakdown_reads_real_alpha_factors_and_sentiment():
    valuation_results = {
        "alpha_factors": {
            "financial": {"Revenue Growth YoY (%)": 12.0},
            "market": {"6-Month Price Momentum (%)": -5.0},
        },
    }
    result = derive_signal_quality(
        _rec(), valuation_results, {}, {"Overall Sentiment": "Positive"}, None,
    )
    breakdown = result["breakdown"]
    assert breakdown["fundamentals"] == "Bullish"
    assert breakdown["momentum"] == "Bearish"
    assert breakdown["sentiment"] == "Bullish"


def test_signal_quality_breakdown_institutional_consensus_majority_vote():
    institutional_consensus = {
        "recommendation_consensus": {
            "institutional_ratings": [
                {"rating": "BUY"}, {"rating": "BUY"}, {"rating": "SELL"},
            ],
        },
    }
    result = derive_signal_quality(_rec(), {}, {}, {}, institutional_consensus)
    assert result["breakdown"]["institutional_consensus"] == "Bullish"
