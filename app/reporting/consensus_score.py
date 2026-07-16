"""
consensus_score.py

Computes the Institutional Consensus Score (ICS): how closely
FinSight's own, independently-derived recommendation agrees with real
institutional analyst ratings for the same ticker. This is an
evaluation metric only -- it is computed from FinSight's
already-finalized recommendation and never feeds back into it (no
caller in this module, or anywhere else, may use its output to alter
context.report_data["recommendation"], valuation, or the narrative
prompt).

v1 scope, deliberately: a straightforward binary match on the
BUY/HOLD/SELL call only. No weighting by firm size/reputation, no
valuation/thesis/risk comparison, no historical tracking -- those are
explicitly future versions (see FUTURE_DIMENSIONS below), and a
simple, fully-statable formula is more defensible for v1 than one
tuned to fit illustrative examples.
"""

from typing import Dict, List, Optional

# Fixed, disclosed thresholds -- shown alongside the label so the
# banding is never a bare, unexplained percentage. Same pattern as
# BUY_THRESHOLD/SELL_THRESHOLD in report_data_builder.py.
STRONG_AGREEMENT_THRESHOLD = 75.0
MODERATE_AGREEMENT_THRESHOLD = 50.0
LOW_AGREEMENT_THRESHOLD = 25.0


def _agreement_label(score: float) -> str:
    if score >= STRONG_AGREEMENT_THRESHOLD:
        return "Strong Agreement"
    if score >= MODERATE_AGREEMENT_THRESHOLD:
        return "Moderate Agreement"
    if score >= LOW_AGREEMENT_THRESHOLD:
        return "Low Market Agreement"
    return "Weak Agreement"


def build_consensus_report(
    finsight_rating: str,
    institutional_ratings: List[Dict],
) -> Optional[Dict]:
    """
    Returns a self-contained consensus report, or None if no real
    institutional ratings are available for this ticker (some stocks
    have no analyst coverage at all) -- callers should render "not
    available" rather than a misleading 0% or 100%.

    finsight_rating: FinSight's own "Buy"/"Hold"/"Sell"/"Insufficient
    Data" recommendation, already finalized before this is called.
    """

    if finsight_rating not in ("Buy", "Hold", "Sell"):
        # "Insufficient Data" (no valuation) has nothing meaningful to
        # compare against institutional calls either.
        return None

    if not institutional_ratings:
        return None

    finsight_upper = finsight_rating.upper()

    agreeing = []
    disagreeing = []

    for rating in institutional_ratings:
        if rating["rating"] == finsight_upper:
            agreeing.append(rating)
        else:
            disagreeing.append(rating)

    total = len(institutional_ratings)
    score = round(len(agreeing) / total * 100)

    return {
        "score": score,
        "label": _agreement_label(score),
        "finsight_rating": finsight_rating,
        "institutional_ratings": institutional_ratings,
        "agreeing_count": len(agreeing),
        "disagreeing_count": len(disagreeing),
        "total_count": total,
        "summary": (
            f"{len(agreeing)} institution{'s' if len(agreeing) != 1 else ''} "
            f"{'agree' if len(agreeing) != 1 else 'agrees'}. "
            f"{len(disagreeing)} institution{'s' if len(disagreeing) != 1 else ''} "
            f"{'disagree' if len(disagreeing) != 1 else 'disagrees'}."
        ),
        "methodology": (
            f"Binary match against FinSight's own {finsight_rating} rating -- "
            f"{STRONG_AGREEMENT_THRESHOLD:.0f}%+ is Strong Agreement, "
            f"{MODERATE_AGREEMENT_THRESHOLD:.0f}-{STRONG_AGREEMENT_THRESHOLD - 1:.0f}% is Moderate, "
            f"{LOW_AGREEMENT_THRESHOLD:.0f}-{MODERATE_AGREEMENT_THRESHOLD - 1:.0f}% is Low Market Agreement, "
            f"below that is Weak Agreement. This score does not influence "
            f"FinSight's recommendation, valuation, or thesis in any way -- "
            f"it is computed after FinSight's own analysis is already final."
        ),
    }
