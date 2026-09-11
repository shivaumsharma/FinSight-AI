"""
Unit tests for FinancialAnalysisBuilder's Revenue CAGR calculation
(app/analysis/financial_analysis.py) -- scoped to the crash this
section had no guard against, not full coverage of the whole module.

Confirmed real failure before the fix: a pre-revenue/SPAC-shaped
history like [0, 0, 2_000_000, 8_000_000] raised ZeroDivisionError
(start=0), and a negative restated year raised TypeError/produced a
complex number under Python's fractional-exponent rule for a negative
base, which then crashed round(). Every sibling CAGR-style calculation
in this codebase (FCFFEngine.calculate_revenue_cagr, growth_metrics.py's
_compute_cagr) already guards start<=0 -- this one didn't.
"""

import pandas as pd

from app.analysis.financial_analysis import FinancialAnalysisBuilder


def _df(revenue):
    index = pd.date_range("2021-12-31", periods=len(revenue), freq="YE")
    return pd.DataFrame({"revenue": revenue}, index=index)


def test_revenue_cagr_unavailable_not_a_crash_when_starting_revenue_is_zero():
    df = _df([0, 0, 2_000_000, 8_000_000])
    summary = FinancialAnalysisBuilder().build(df)
    assert summary["Revenue CAGR (%)"] == "Unavailable"


def test_revenue_cagr_unavailable_not_a_crash_when_starting_revenue_is_negative():
    # A restated/reversed-revenue year -- (end/start) with a negative
    # denominator raised to a fractional exponent is a complex number
    # in Python, not just a bad float; must degrade the same as the
    # zero-start case, not raise.
    df = _df([-500_000, 100_000, 2_000_000, 8_000_000])
    summary = FinancialAnalysisBuilder().build(df)
    assert summary["Revenue CAGR (%)"] == "Unavailable"


def test_revenue_cagr_unavailable_when_ending_revenue_is_zero_or_negative():
    df = _df([2_000_000, 1_000_000, 500_000, 0])
    summary = FinancialAnalysisBuilder().build(df)
    assert summary["Revenue CAGR (%)"] == "Unavailable"


def test_revenue_cagr_computes_normally_for_an_ordinary_growth_history():
    # Regression check: the fix must not have broken the ordinary case.
    # 4 points -> 3 years, 100 -> 133.1 is a clean 10%/yr CAGR.
    df = _df([100, 110, 121, 133.1])
    summary = FinancialAnalysisBuilder().build(df)
    assert summary["Revenue CAGR (%)"] == 10.0


def test_revenue_cagr_unavailable_with_fewer_than_three_data_points():
    df = _df([100, 110])
    summary = FinancialAnalysisBuilder().build(df)
    assert summary["Revenue CAGR (%)"] == "Unavailable"
