"""
Unit tests for DCFEngine (app/valuation/dcf_engine.py) -- terminal
value, discounting, and the WACC floor that keeps the Gordon-growth
term from blowing up when discount_rate sits too close to
terminal_growth_rate.
"""

import pandas as pd
import pytest

from app.valuation.dcf_engine import DCFEngine


def _forecast(values):
    index = [f"Year_{i+1}" for i in range(len(values))]
    return pd.DataFrame({"forecast_fcff": values}, index=index)


def test_wacc_is_not_floored_when_spread_is_wide_enough():
    engine = DCFEngine(_forecast([100]), discount_rate=0.10, terminal_growth_rate=0.04)
    assert engine.wacc_floored is False
    assert engine.discount_rate == pytest.approx(0.10)


def test_wacc_is_floored_when_spread_is_too_tight():
    # terminal_growth=0.04, MIN_WACC_TERMINAL_SPREAD=0.03 -> floor is 0.07.
    # A raw WACC of 0.05 is only a 1pt spread, well under the floor.
    engine = DCFEngine(_forecast([100]), discount_rate=0.05, terminal_growth_rate=0.04)
    assert engine.wacc_floored is True
    assert engine.discount_rate == pytest.approx(0.04 + DCFEngine.MIN_WACC_TERMINAL_SPREAD)


def test_wacc_floor_info_preserves_the_raw_unfloored_value():
    engine = DCFEngine(_forecast([100]), discount_rate=0.05, terminal_growth_rate=0.04)
    info = engine.wacc_floor_info()
    assert info["raw_wacc"] == pytest.approx(0.05)
    assert info["wacc_used"] == pytest.approx(0.07)
    assert info["floored"] is True


def test_discount_factors_compound_correctly():
    engine = DCFEngine(_forecast([100, 100, 100]), discount_rate=0.10, terminal_growth_rate=0.02)
    factors = engine.calculate_discount_factors()
    assert factors.iloc[0] == pytest.approx(1 / 1.10)
    assert factors.iloc[1] == pytest.approx(1 / 1.10 ** 2)
    assert factors.iloc[2] == pytest.approx(1 / 1.10 ** 3)


def test_terminal_value_uses_gordon_growth_on_final_year():
    engine = DCFEngine(_forecast([100, 110]), discount_rate=0.10, terminal_growth_rate=0.03)
    final_fcff = 110
    expected = (final_fcff * 1.03) / (0.10 - 0.03)
    assert engine.calculate_terminal_value() == pytest.approx(expected)


def test_enterprise_value_is_pv_of_fcff_plus_pv_of_terminal_value():
    engine = DCFEngine(_forecast([100, 100]), discount_rate=0.10, terminal_growth_rate=0.03)
    pv_fcff = 100 / 1.10 + 100 / 1.10 ** 2
    pv_terminal = engine.discount_terminal_value()
    assert engine.calculate_enterprise_value() == pytest.approx(pv_fcff + pv_terminal)


def test_equity_value_subtracts_net_debt():
    engine = DCFEngine(_forecast([100]), discount_rate=0.10, terminal_growth_rate=0.03)
    ev = engine.calculate_enterprise_value()
    equity_value = engine.calculate_equity_value(total_debt=500, cash=200)
    assert equity_value == pytest.approx(ev - (500 - 200))


def test_intrinsic_value_divides_equity_value_by_shares():
    engine = DCFEngine(_forecast([100]), discount_rate=0.10, terminal_growth_rate=0.03)
    equity_value = engine.calculate_equity_value(total_debt=0, cash=0)
    intrinsic = engine.calculate_intrinsic_value(shares_outstanding=1000, total_debt=0, cash=0)
    assert intrinsic == pytest.approx(equity_value / 1000)
