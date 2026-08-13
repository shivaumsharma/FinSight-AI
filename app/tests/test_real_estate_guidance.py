"""
Unit tests for app/reasoning/real_estate_guidance.py -- pure function
over a plain user dict, no mocking needed.
"""

from app.reasoning import real_estate_guidance as reg


def test_returns_none_when_not_interested():
    user = {"interested_in_real_estate": 0, "risk_tolerance": "Moderate"}
    assert reg.get_real_estate_guidance(user) is None


def test_returns_none_for_a_missing_user():
    assert reg.get_real_estate_guidance(None) is None


def test_returns_guidance_when_interested():
    user = {"interested_in_real_estate": 1, "risk_tolerance": "Aggressive"}
    result = reg.get_real_estate_guidance(user)
    assert result["target_allocation_low_pct"] == 5.0
    assert result["target_allocation_high_pct"] == 10.0
    assert result["example_reit_tickers"] == ["VNQ", "O", "PLD"]
    assert "not a recommendation" in result["disclaimer"].lower()


def test_conservative_gets_a_higher_target_than_aggressive():
    conservative = reg.get_real_estate_guidance({"interested_in_real_estate": 1, "risk_tolerance": "Conservative"})
    aggressive = reg.get_real_estate_guidance({"interested_in_real_estate": 1, "risk_tolerance": "Aggressive"})
    assert conservative["target_allocation_low_pct"] > aggressive["target_allocation_low_pct"]


def test_missing_risk_tolerance_defaults_to_moderate():
    user = {"interested_in_real_estate": 1, "risk_tolerance": None}
    result = reg.get_real_estate_guidance(user)
    moderate = reg.get_real_estate_guidance({"interested_in_real_estate": 1, "risk_tolerance": "Moderate"})
    assert result["target_allocation_low_pct"] == moderate["target_allocation_low_pct"]
