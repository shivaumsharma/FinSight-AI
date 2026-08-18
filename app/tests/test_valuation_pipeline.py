"""
Unit tests for ValuationPipeline's leverage-instability guard
(app/valuation/valuation_pipeline.py, MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO)
-- when net debt consumes nearly all (or more than) the DCF's own
computed enterprise value, equity_value = enterprise_value - net_debt
becomes a small, unstable difference between two much larger,
individually uncertain numbers, not a reliable per-share value.

Confirmed on two real companies (KHC, HAS) via scripts/wacc_capm_audit.py:
neither has negative FCFF or a broken share count -- the near-zero
result comes purely from this leverage-extraction step. See the
MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO comment in valuation_pipeline.py
for the full calibration evidence (a stark, unambiguous gap across the
~80-ticker backtest universe: the three real failures sit at -56.1%/
0.9%/2.4%, every legitimate company at 44.1%+).
"""

import pandas as pd
import pytest

from app.valuation.valuation_pipeline import (
    ValuationPipeline,
    MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO,
    RISK_TOLERANCE_WACC_ADJUSTMENT_PCT,
)


def _financial_df(total_debt, cash):
    index = pd.to_datetime(["2023-12-31", "2024-12-31"])
    return pd.DataFrame({
        "revenue": [100, 110], "ebit": [20, 22], "net_income": [15, 16],
        "cash_from_operations": [18, 19], "capex": [5, 5],
        "total_debt": total_debt, "tax_expense": [3, 3], "pretax_income": [18, 19],
        "depreciation": [4, 4], "current_assets": [30, 32], "current_liabilities": [20, 21],
        "cash": cash, "shares_outstanding": [1000, 1000], "interest_expense": [-2, -2],
        "total_equity": [80, 85],
    }, index=index)


def test_valuation_stays_available_for_reasonable_leverage():
    # Baseline sanity check -- a normally-levered company (debt well
    # under the enterprise value a DCF this size would compute) must
    # not trip the new guard.
    df = _financial_df(total_debt=[50, 55], cash=[40, 42])
    result = ValuationPipeline(financial_df=df, market_cap=5000, beta=1.1, ticker="TEST").run_valuation()
    assert result["dcf_available"] is True
    assert result["equity_value"] > 0


def test_valuation_unavailable_when_net_debt_swamps_enterprise_value():
    # Debt scaled to dramatically exceed anything this size of company
    # could plausibly discount to -- KHC/HAS-shaped failure, just with
    # a wide enough margin that the exact WACC/FCFF arithmetic doesn't
    # need to be hand-computed to guarantee the guard fires.
    df = _financial_df(total_debt=[500_000, 500_000], cash=[5, 5])
    result = ValuationPipeline(financial_df=df, market_cap=5000, beta=1.1, ticker="TEST").run_valuation()

    assert result["dcf_available"] is False
    assert result["enterprise_value"] is None
    assert result["equity_value"] is None
    assert "net debt" in result["dcf_unavailable_reason"].lower()
    assert f"{MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO:.0%}" in result["dcf_unavailable_reason"]


def test_valuation_unavailable_when_debt_literally_exceeds_enterprise_value():
    # The CVNA case: equity_value comes out negative, not just thin.
    # Must be caught the same way, not treated as "somehow still fine
    # because the ratio is technically defined."
    df = _financial_df(total_debt=[1_000_000, 1_000_000], cash=[0, 0])
    result = ValuationPipeline(financial_df=df, market_cap=5000, beta=1.1, ticker="TEST").run_valuation()
    assert result["dcf_available"] is False


def test_valuation_unavailable_when_market_cap_is_none():
    # context.market_cap can be None (market_data_tool.py only guards
    # company_name/current_price before populating it from yfinance's
    # own company_info, which can omit marketCap) -- WACCEngine uses
    # market_cap directly in arithmetic (equity_value+debt_value,
    # equity_value/total_value) that would raise a TypeError against
    # None rather than degrade. Must route to the same unavailable-DCF
    # path as every other structurally-missing-input case, not crash.
    df = _financial_df(total_debt=[50, 55], cash=[40, 42])
    result = ValuationPipeline(financial_df=df, market_cap=None, beta=1.1, ticker="TEST").run_valuation()

    assert result["dcf_available"] is False
    assert result["enterprise_value"] is None
    assert result["equity_value"] is None
    assert "market capitalization" in result["dcf_unavailable_reason"].lower()


# ---------------------------------------------------------------- risk_tolerance -> WACC/intrinsic value
#
# See RISK_TOLERANCE_WACC_ADJUSTMENT_PCT's own comment in
# valuation_pipeline.py: a user's saved risk_tolerance shifts
# self.risk_free_rate (added, not replaced) before WACCEngine computes
# cost of equity, so it should move WACC -- and therefore intrinsic
# value -- in a specific, testable direction: Conservative UP (more
# cautious => lower intrinsic value), Aggressive DOWN (more optimistic
# => higher intrinsic value), Moderate unchanged (the DB's own default,
# a deliberate no-op for the common case).

def _risk_tolerance_financial_df():
    # Deliberately well inside every other guard in this file (modest
    # leverage, positive base FCFF, computable WACC) so the only thing
    # that can possibly move between the three risk_tolerance runs below
    # is the risk_free_rate adjustment itself, not some unrelated guard
    # tripping differently across runs.
    index = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])
    return pd.DataFrame({
        "revenue": [100, 110, 121], "ebit": [20, 22, 24], "net_income": [15, 16, 17],
        "cash_from_operations": [18, 19, 20], "capex": [5, 5, 5],
        "total_debt": [50, 52, 55], "tax_expense": [3, 3, 3], "pretax_income": [18, 19, 20],
        "depreciation": [4, 4, 4], "current_assets": [30, 32, 34], "current_liabilities": [20, 21, 22],
        "cash": [40, 41, 42], "shares_outstanding": [1000, 1000, 1000], "interest_expense": [-2, -2, -2],
        "total_equity": [80, 85, 90],
    }, index=index)


def _run_with_risk_tolerance(risk_tolerance=None):
    df = _risk_tolerance_financial_df()
    kwargs = dict(financial_df=df, market_cap=5000, beta=1.1, ticker="RISKTEST")
    if risk_tolerance is not None:
        kwargs["risk_tolerance"] = risk_tolerance
    return ValuationPipeline(**kwargs).run_valuation()


def test_moderate_risk_tolerance_matches_no_parameter_call_exactly():
    # The no-regression check: a caller that doesn't pass risk_tolerance
    # at all (every caller before this feature existed) must get
    # byte-for-byte the same result as a caller that explicitly passes
    # "Moderate" -- both correspond to a 0.0 risk_free_rate adjustment.
    assert RISK_TOLERANCE_WACC_ADJUSTMENT_PCT["Moderate"] == 0.0

    result_default = _run_with_risk_tolerance(None)
    result_moderate = _run_with_risk_tolerance("Moderate")

    assert result_default["dcf_available"] is True
    assert result_default["wacc"] == result_moderate["wacc"]
    assert result_default["raw_wacc"] == result_moderate["raw_wacc"]
    assert result_default["enterprise_value"] == result_moderate["enterprise_value"]
    assert result_default["equity_value"] == result_moderate["equity_value"]


def test_conservative_vs_moderate_vs_aggressive_move_wacc_and_value_in_expected_direction():
    conservative = _run_with_risk_tolerance("Conservative")
    moderate = _run_with_risk_tolerance("Moderate")
    aggressive = _run_with_risk_tolerance("Aggressive")

    for result in (conservative, moderate, aggressive):
        assert result["dcf_available"] is True

    # Conservative = higher discount rate (higher raw_wacc) = more
    # cautious = LOWER intrinsic value (equity_value, a direct proxy for
    # intrinsic value per share here since shares_outstanding is fixed
    # across all three runs).
    assert conservative["raw_wacc"] > moderate["raw_wacc"] > aggressive["raw_wacc"]
    assert conservative["equity_value"] < moderate["equity_value"] < aggressive["equity_value"]

    # The move should be exactly the configured adjustment (0.015 either
    # direction), not some other coincidental drift -- WACC is a
    # market-value-weighted blend of cost of equity and cost of debt
    # (see WACCEngine.calculate_wacc), and only the cost-of-equity leg
    # moves here, so the raw_wacc delta is the risk_free_rate delta
    # scaled down by the equity weight (equity_value / total_value) < 1,
    # not the full 1.5pp -- assert the direction and a sane upper bound
    # instead of an exact figure that would double as a second
    # implementation of WACCEngine's own math.
    adjustment = RISK_TOLERANCE_WACC_ADJUSTMENT_PCT["Conservative"] - RISK_TOLERANCE_WACC_ADJUSTMENT_PCT["Aggressive"]
    assert 0 < (conservative["raw_wacc"] - aggressive["raw_wacc"]) <= adjustment + 1e-9
