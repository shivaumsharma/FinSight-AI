"""
Unit tests for MonteCarloDCFEngine (app/valuation/monte_carlo_dcf.py) --
samples growth/WACC/terminal-growth around a base case and prices each
sample through the real FCFFEngine/DCFEngine (not mocked; those are
FinSight's own well-tested valuation math, the true "function under
test" boundary is only numpy's RNG). Covers: graceful degradation to
an empty array when the base FCFF can't be computed (same class of
input as fcff_engine.py's single-data-point guard), the unstable-
sample rejection filter (WACC/terminal-growth spread below
DCFEngine.MIN_WACC_TERMINAL_SPREAD), reproducibility under a fixed
seed, and internal consistency of the summary statistics.
"""

import numpy as np
import pandas as pd
import pytest

from app.valuation.dcf_engine import DCFEngine
from app.valuation.fcff_engine import FCFFEngine
from app.valuation.monte_carlo_dcf import MonteCarloDCFEngine


def _financial_df(rows=2):
    """A small, internally-consistent multi-year financial_df -- same
    shape as test_valuation_pipeline.py's fixture, confirmed to
    produce a usable (positive) normalized base FCFF."""
    base = {
        "revenue": [100, 110], "ebit": [20, 22], "net_income": [15, 16],
        "cash_from_operations": [18, 19], "capex": [5, 5],
        "total_debt": [50, 55], "tax_expense": [3, 3], "pretax_income": [18, 19],
        "depreciation": [4, 4], "current_assets": [30, 32], "current_liabilities": [20, 21],
        "cash": [40, 42], "shares_outstanding": [1000, 1000], "interest_expense": [-2, -2],
        "total_equity": [80, 85],
    }
    index = pd.to_datetime(["2023-12-31", "2024-12-31"])
    if rows == 1:
        base = {k: [v[-1]] for k, v in base.items()}
        index = index[-1:]
    return pd.DataFrame(base, index=index)


def _engine(financial_df, **overrides):
    fcff_engine = FCFFEngine(financial_df)
    kwargs = dict(
        fcff_engine=fcff_engine,
        base_growth_rate=fcff_engine.calculate_revenue_cagr() or 0.05,
        base_wacc=0.10,
        base_terminal_growth=0.03,
        total_debt=financial_df["total_debt"].iloc[-1],
        cash=financial_df["cash"].iloc[-1],
        shares_outstanding=financial_df["shares_outstanding"].iloc[-1],
        iterations=500,
        random_state=42,
    )
    kwargs.update(overrides)
    return MonteCarloDCFEngine(**kwargs)


# ---------------------------------------------------------------- run(): graceful degradation

def test_run_returns_empty_array_when_base_fcff_is_negative():
    df = _financial_df()
    # Force a negative normalized base FCFF the way valuation_pipeline.py's
    # own docstring describes (working-capital consumption/capex outstrips
    # profitability), rather than an unrelated structural failure.
    df["capex"] = [500, 500]
    fcff_engine = FCFFEngine(df)
    assert fcff_engine.calculate_normalized_base_fcff() < 0

    mc = _engine(df, fcff_engine=fcff_engine, base_growth_rate=0.05)
    result = mc.run()
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_run_returns_empty_array_when_base_fcff_is_none():
    # A single-fiscal-year financial_df: calculate_normalized_base_fcff()
    # returns None because the change-in-NWC ratio has no prior year to
    # diff against (empty ratio series -> NaN average) -- confirmed
    # directly against FCFFEngine, the same "single-data-point revenue
    # series" class of guard this session added to fcff_engine.py.
    df = _financial_df(rows=1)
    fcff_engine = FCFFEngine(df)
    assert fcff_engine.calculate_normalized_base_fcff() is None

    mc = _engine(df, fcff_engine=fcff_engine, base_growth_rate=0.05)
    result = mc.run()
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


# ---------------------------------------------------------------- run(): sampling behavior

def test_run_rejects_samples_whose_wacc_terminal_spread_is_too_tight():
    """Deterministic check of the rejection filter itself: feed a fixed
    sequence of growth/WACC/terminal samples (bypassing real
    randomness, the true external seam) and confirm exactly the
    samples with (wacc - terminal) < DCFEngine.MIN_WACC_TERMINAL_SPREAD
    are dropped, not floored."""

    class _SequenceRNG:
        def __init__(self, growth, wacc, terminal):
            self._arrays = [np.array(growth), np.array(wacc), np.array(terminal)]
            self._i = 0

        def normal(self, loc, scale, size):
            arr = self._arrays[self._i]
            self._i += 1
            return arr

    df = _financial_df()
    mc = _engine(df, iterations=3)
    # spreads: 0.10-0.03=0.07 (valid), 0.05-0.04=0.01 (invalid, < 0.03),
    # 0.12-0.03=0.09 (valid) -> exactly 2 of 3 samples should survive.
    mc._rng = _SequenceRNG(
        growth=[0.05, 0.05, 0.05],
        wacc=[0.10, 0.05, 0.12],
        terminal=[0.03, 0.04, 0.03],
    )
    assert (0.05 - 0.04) < DCFEngine.MIN_WACC_TERMINAL_SPREAD

    result = mc.run()
    assert len(result) == 2


def test_run_produces_genuinely_varying_outcomes_not_a_repeated_point_estimate():
    df = _financial_df()
    mc = _engine(df)
    result = mc.run()
    assert len(result) > 0
    # Not every sample collapsed onto the same value -- this is a
    # distribution, not DCFEngine's point estimate repeated N times.
    assert np.std(result) > 0
    assert len(np.unique(result)) > 1


def test_run_is_reproducible_under_a_fixed_random_state():
    df = _financial_df()
    result_a = _engine(df, random_state=123).run()
    result_b = _engine(df, random_state=123).run()
    np.testing.assert_array_equal(result_a, result_b)


def test_run_differs_across_different_random_states():
    df = _financial_df()
    result_a = _engine(df, random_state=1).run()
    result_b = _engine(df, random_state=2).run()
    assert not np.array_equal(result_a, result_b)


# ---------------------------------------------------------------- statistics()

def test_statistics_returns_none_for_none_input():
    assert MonteCarloDCFEngine.statistics(None, current_price=100) is None


def test_statistics_returns_none_for_empty_array():
    assert MonteCarloDCFEngine.statistics(np.array([]), current_price=100) is None


def test_statistics_computes_expected_values_on_hand_built_distribution():
    # 9 evenly-spaced values -- percentiles land on clean, hand-checkable
    # numbers (numpy's default linear interpolation).
    values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    stats = MonteCarloDCFEngine.statistics(values, current_price=45.0)

    assert stats["mean"] == pytest.approx(50.0)
    assert stats["median"] == pytest.approx(50.0)
    assert stats["std_dev"] == pytest.approx(np.std(values))
    assert stats["p25"] == pytest.approx(30.0)
    assert stats["p75"] == pytest.approx(70.0)
    assert stats["ci_lower"] == pytest.approx(14.0)   # 5th percentile
    assert stats["ci_upper"] == pytest.approx(86.0)   # 95th percentile
    # values > 45: 50,60,70,80,90 -> 5 of 9
    assert stats["prob_undervalued"] == pytest.approx(5 / 9)
    assert stats["n_samples"] == 9


def test_statistics_are_internally_consistent_on_real_run_output():
    df = _financial_df()
    mc = _engine(df, iterations=800, random_state=7)
    mc_values = mc.run()
    stats = MonteCarloDCFEngine.statistics(mc_values, current_price=50.0)

    assert stats is not None
    assert stats["ci_lower"] <= stats["p25"] <= stats["median"] <= stats["p75"] <= stats["ci_upper"]
    assert 0.0 <= stats["prob_undervalued"] <= 1.0
    assert stats["std_dev"] >= 0
    assert stats["n_samples"] == len(mc_values)
