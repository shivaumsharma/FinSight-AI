"""
real_estate_guidance.py

Powers the Profile page's "Real Estate" card -- allocation-level
guidance only (a target % of portfolio, keyed off risk tolerance) plus
a short, clearly-labeled list of liquid REIT tickers to look up on the
existing stock detail page. Deliberately NOT property-specific
recommendations -- no free data source exists for property-level
valuation, the same gap as shareholding-pattern data elsewhere in this
app (see the approved onboarding-questionnaire plan's real-estate scope
note). REITs themselves need no new data plumbing: they're ordinary,
exchange-traded equities already served by the existing stock detail
page, ValuationTool included.
"""

from typing import Optional

# (low, high) target allocation % -- a plain lookup table, same
# breakpoint-table style as stock_score.py's threshold helpers, not a
# personalized calculation. Conventional diversification ranges, not
# derived from this user's actual portfolio composition.
_ALLOCATION_BY_RISK = {
    "Conservative": (15.0, 20.0),
    "Moderate": (10.0, 15.0),
    "Aggressive": (5.0, 10.0),
}

# Large, liquid, well-known REITs -- examples to look up, not a buy
# recommendation (see the disclaimer this ships alongside below).
EXAMPLE_REIT_TICKERS = ["VNQ", "O", "PLD"]


def get_real_estate_guidance(user: Optional[dict]) -> Optional[dict]:
    """None if the user hasn't opted into real-estate interest during
    onboarding (or doesn't exist) -- the Profile card simply doesn't
    render for anyone else. `user` is db.get_user_by_id()'s own row
    shape (a plain dict, SQLite booleans as 0/1)."""
    if not user or not user.get("interested_in_real_estate"):
        return None

    risk_tolerance = user.get("risk_tolerance") or "Moderate"
    low, high = _ALLOCATION_BY_RISK.get(risk_tolerance, _ALLOCATION_BY_RISK["Moderate"])

    return {
        "target_allocation_low_pct": low,
        "target_allocation_high_pct": high,
        "example_reit_tickers": EXAMPLE_REIT_TICKERS,
        "disclaimer": (
            "Informational only, not a recommendation to buy these specific securities or any "
            "property -- consult a licensed financial advisor before acting on this."
        ),
    }
