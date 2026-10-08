"""
Tests for app/analysis/portfolio_metrics.py -- every function checked
against a small synthetic case with a hand-computable expected value,
not just "does it run without raising."
"""

import numpy as np
import pandas as pd
import pytest

from app.analysis import portfolio_metrics as pm


def _curve(values, start="2023-01-01", freq="365D"):
    index = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=index)


def test_periodic_returns_is_simple_pct_change():
    curve = _curve([100.0, 110.0, 99.0])
    returns = pm.periodic_returns(curve)
    assert returns.tolist() == pytest.approx([0.10, -0.10])


def test_cagr_of_a_doubling_over_exactly_two_years():
    # 100 -> 121 over exactly 2 years (365.25*2 days) is a clean 10%/yr
    # compound rate: 1.10^2 = 1.21.
    index = pd.DatetimeIndex(["2020-01-01", "2022-01-01"])
    curve = pd.Series([100.0, 121.0], index=index)
    assert pm.cagr(curve) == pytest.approx(0.10, abs=0.002)


def test_cagr_none_for_a_single_point_or_non_positive_start():
    assert pm.cagr(_curve([100.0])) is None
    assert pm.cagr(_curve([0.0, 50.0])) is None
    assert pm.cagr(_curve([-10.0, 50.0])) is None


def test_annualized_volatility_matches_hand_computed_stdev():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    expected = returns.std() * np.sqrt(4)
    assert pm.annualized_volatility(returns, periods_per_year=4) == pytest.approx(expected)


def test_sharpe_ratio_zero_risk_free_matches_mean_over_std():
    returns = pd.Series([0.05, 0.02, -0.01, 0.03])
    expected = (returns.mean() / returns.std()) * np.sqrt(4)
    assert pm.sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=4) == pytest.approx(expected)


def test_sharpe_ratio_none_for_zero_variance_returns():
    returns = pd.Series([0.02, 0.02, 0.02, 0.02])
    assert pm.sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=4) is None


def test_sortino_ratio_zero_mean_excess_is_exactly_zero():
    # Symmetric gains/losses -> mean excess return is exactly 0, so
    # Sortino is 0 regardless of the downside deviation's magnitude --
    # a clean, hand-checkable case independent of the downside formula.
    symmetric = pd.Series([0.05, -0.05, 0.05, -0.05])
    assert pm.sortino_ratio(symmetric, risk_free_rate=0.0, periods_per_year=4) == pytest.approx(0.0)


def test_sortino_ratio_penalizes_larger_downside_more_at_equal_mean():
    # Both series have the identical mean excess return (0.01) but
    # small_downside's losing periods are smaller in magnitude than
    # large_downside's -- Sortino must rank small_downside higher
    # (less risk for the same reward), which Sharpe alone wouldn't
    # necessarily do since these two also have different FULL stdev.
    small_downside = pd.Series([0.03, -0.01, 0.03, -0.01])  # mean = 0.01
    large_downside = pd.Series([0.05, -0.03, 0.03, -0.01])  # mean = 0.01
    assert small_downside.mean() == pytest.approx(large_downside.mean())

    sortino_small = pm.sortino_ratio(small_downside, risk_free_rate=0.0, periods_per_year=4)
    sortino_large = pm.sortino_ratio(large_downside, risk_free_rate=0.0, periods_per_year=4)
    assert sortino_small > sortino_large > 0


def test_sortino_ratio_none_when_no_downside_periods_exist():
    all_gains = pd.Series([0.01, 0.02, 0.03])
    assert pm.sortino_ratio(all_gains, risk_free_rate=0.0, periods_per_year=4) is None


def test_max_drawdown_of_a_known_peak_and_trough():
    # Peaks at 120, troughs at 90 before partially recovering to 110 --
    # the worst peak-to-trough move is 90/120 - 1 = -25%, not the
    # smaller 110-vs-120 or 110-vs-90 comparisons.
    curve = _curve([100.0, 120.0, 90.0, 110.0])
    assert pm.max_drawdown(curve) == pytest.approx(-0.25)


def test_max_drawdown_is_zero_for_a_monotonically_rising_curve():
    curve = _curve([100.0, 105.0, 110.0, 120.0])
    assert pm.max_drawdown(curve) == pytest.approx(0.0)


def test_hit_rate_counts_strictly_positive_periods_only():
    # Exactly 2 of 5 are strictly positive -- the zero is neither a
    # hit nor a miss under a strict ">0" rule, so it must not inflate
    # the rate to 3/5.
    returns = pd.Series([0.05, -0.02, 0.03, -0.01, 0.0])
    assert pm.hit_rate(returns) == pytest.approx(2 / 5)


def test_profit_factor_ratio_of_gross_gains_to_gross_losses():
    returns = pd.Series([0.10, 0.05, -0.02, -0.03])
    expected = (0.10 + 0.05) / (0.02 + 0.03)
    assert pm.profit_factor(returns) == pytest.approx(expected)


def test_profit_factor_none_when_there_are_no_losing_periods():
    all_gains = pd.Series([0.01, 0.02])
    assert pm.profit_factor(all_gains) is None


def test_turnover_of_a_full_portfolio_replacement_is_one_not_two():
    previous = {"AAPL": 0.5, "MSFT": 0.5}
    target = {"GOOGL": 0.5, "AMZN": 0.5}
    assert pm.turnover(previous, target) == pytest.approx(1.0)


def test_turnover_of_an_unchanged_portfolio_is_zero():
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    assert pm.turnover(weights, dict(weights)) == pytest.approx(0.0)


def test_turnover_of_a_partial_rebalance():
    # AAPL trimmed 0.6->0.4 (0.2 sold), MSFT added 0.0->0.2 (0.2 bought)
    # -- total absolute change is 0.4, one-sided turnover is 0.2.
    previous = {"AAPL": 0.6, "GOOGL": 0.4}
    target = {"AAPL": 0.4, "GOOGL": 0.4, "MSFT": 0.2}
    assert pm.turnover(previous, target) == pytest.approx(0.2)
