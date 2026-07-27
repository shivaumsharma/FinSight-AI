"""
Unit tests for the recommendation formula (app/reporting/report_data_builder.py)
-- the single most consequential piece of deterministic logic in the
project, since it decides Buy/Hold/Sell before the LLM ever writes a
word. Includes a regression test for the NaN-to-+100 bug: Python's
`nan < x` is always False, so an un-guarded `max(-100, min(100, nan))`
silently resolves to +100 ("maximally confident Buy") instead of
propagating as unknown -- this must never happen again.
"""

import math

import pytest

from app.reporting.report_data_builder import (
    BUY_THRESHOLD, SELL_THRESHOLD, DCF_WEIGHT, RELATIVE_WEIGHT, SCORE_CAP,
    _dcf_score, _relative_score, _composite_score, _rating_from_score,
    compute_signal_agreement, derive_recommendation,
)


# ---------------------------------------------------------------- _dcf_score

def test_dcf_score_passes_through_moderate_upside():
    assert _dcf_score(12.5) == 12.5


@pytest.mark.parametrize("upside,expected", [(500, SCORE_CAP), (-500, -SCORE_CAP)])
def test_dcf_score_is_capped(upside, expected):
    assert _dcf_score(upside) == expected


# ----------------------------------------------------------- _relative_score

def test_relative_score_flips_sign_of_vs_history_pct():
    # relative_valuation.py: positive vs_history_pct = expensive (bearish).
    # _relative_score puts everything on the same bullish-positive scale.
    assert _relative_score({"vs_history_pct": 20.0}) == -20.0
    assert _relative_score({"vs_history_pct": -20.0}) == 20.0


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

def _valuation(upside_percent, relative_valuation=None, dcf_available=True):
    return {
        "upside_percent": upside_percent,
        "relative_valuation": relative_valuation,
        "dcf_available": dcf_available,
        "dcf_unavailable_reason": None if dcf_available else "test reason",
    }


def test_derive_recommendation_buy_from_strong_upside():
    result = derive_recommendation(_valuation(upside_percent=50.0))
    assert result["rating"] == "Buy"
    assert result["dcf_score"] == 50.0
    assert result["composite_score"] == 50.0  # no relative valuation -> 100% DCF weight


def test_derive_recommendation_sell_from_strong_downside():
    result = derive_recommendation(_valuation(upside_percent=-50.0))
    assert result["rating"] == "Sell"


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
