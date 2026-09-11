"""
Unit tests for DDMEngine (app/valuation/ddm_engine.py) -- Gordon Growth
DDM, scoped to consistent, materially-dividend-paying companies only.
Zero prior coverage for this module.
"""

import pandas as pd
import pytest

from app.valuation.ddm_engine import DDMEngine


def _df(**cols):
    n = len(next(iter(cols.values())))
    index = pd.to_datetime([f"20{20+i}-12-31" for i in range(n)])
    return pd.DataFrame(cols, index=index)


# ---------------------------------------------------------------- calculate_dps_series

def test_dps_series_divides_dividends_by_shares_outstanding():
    df = _df(dividends_paid=[100, 110], shares_outstanding=[50, 50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    dps = engine.calculate_dps_series()
    assert list(dps) == pytest.approx([2.0, 2.2])


def test_dps_series_is_empty_when_columns_are_missing():
    df = _df(revenue=[100, 110])
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert engine.calculate_dps_series().empty


def test_dps_series_excludes_a_zero_shares_outstanding_year():
    # A year with shares_outstanding<=0 is structurally missing data,
    # not a real 0 -- must be dropped, not divided into.
    df = _df(dividends_paid=[100, 110], shares_outstanding=[0, 50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    dps = engine.calculate_dps_series()
    assert len(dps) == 1
    assert dps.iloc[0] == pytest.approx(2.2)


# ---------------------------------------------------------------- is_dividend_payer

def test_is_dividend_payer_true_for_a_consistent_material_payer():
    df = _df(
        dividends_paid=[100, 110, 120],
        shares_outstanding=[50, 50, 50],
        net_income=[500, 550, 600],
    )
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert engine.is_dividend_payer()

def test_is_dividend_payer_false_with_fewer_than_min_years():
    df = _df(dividends_paid=[100], shares_outstanding=[50], net_income=[500])
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert not engine.is_dividend_payer()


def test_is_dividend_payer_false_if_any_year_paid_nothing():
    # A dividend that was suspended or only just started -- no durable
    # multi-year track record for the growth extrapolation to mean
    # anything.
    df = _df(
        dividends_paid=[0, 110, 120],
        shares_outstanding=[50, 50, 50],
        net_income=[500, 550, 600],
    )
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert not engine.is_dividend_payer()


def test_is_dividend_payer_false_for_a_token_payout_ratio():
    # Confirmed-live NVDA case from the module's own docstring: years
    # paying and per-year positivity both clear, but the payout is
    # under 15% of net income -- a token gesture, not a real policy.
    df = _df(
        dividends_paid=[10, 11, 12],
        shares_outstanding=[50, 50, 50],
        net_income=[5000, 5500, 6000],
    )
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert not engine.is_dividend_payer()


def test_is_dividend_payer_false_when_net_income_is_missing():
    df = _df(dividends_paid=[100, 110], shares_outstanding=[50, 50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert not engine.is_dividend_payer()


def test_is_dividend_payer_false_when_latest_net_income_is_non_positive():
    df = _df(
        dividends_paid=[100, 110],
        shares_outstanding=[50, 50],
        net_income=[500, -50],
    )
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert not engine.is_dividend_payer()


# ---------------------------------------------------------------- calculate_dps_cagr

def test_dps_cagr_computes_normal_growth():
    df = _df(dividends_paid=[100, 105], shares_outstanding=[50, 50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    # DPS: 2.0 -> 2.1, 1 period -> 5%, below the 6% cap so uncapped.
    assert engine.calculate_dps_cagr() == pytest.approx(0.05)


def test_dps_cagr_caps_at_max_sustainable_growth():
    # Confirmed-live MSFT case: a real but temporary acceleration must
    # not feed straight into a perpetuity.
    df = _df(dividends_paid=[100, 200], shares_outstanding=[50, 50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert engine.calculate_dps_cagr() == pytest.approx(DDMEngine.MAX_SUSTAINABLE_DPS_GROWTH)


def test_dps_cagr_is_none_for_a_single_data_point():
    df = _df(dividends_paid=[100], shares_outstanding=[50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert engine.calculate_dps_cagr() is None


def test_dps_cagr_is_none_for_a_non_positive_beginning_value():
    df = _df(dividends_paid=[0, 100], shares_outstanding=[50, 50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert engine.calculate_dps_cagr() is None


def test_dps_cagr_is_negative_and_uncapped_for_a_declining_payout():
    # The MAX_SUSTAINABLE_DPS_GROWTH cap is a ceiling, not a floor --
    # min(raw_cagr, cap) must let a real decline through unchanged.
    df = _df(dividends_paid=[200, 100], shares_outstanding=[50, 50])
    engine = DDMEngine(df, cost_of_equity=0.08)
    assert engine.calculate_dps_cagr() == pytest.approx(-0.5)


# ---------------------------------------------------------------- calculate_intrinsic_value

def test_intrinsic_value_gordon_growth_happy_path():
    df = _df(
        dividends_paid=[100, 110, 121],
        shares_outstanding=[50, 50, 50],
        net_income=[500, 550, 600],
    )
    engine = DDMEngine(df, cost_of_equity=0.10)
    # DPS 2.0 -> 2.2 -> 2.42, raw cagr=10% but capped at
    # MAX_SUSTAINABLE_DPS_GROWTH=6%. cost_of_equity(0.10) - growth(0.06)
    # = 0.04 >= MIN_COE_GROWTH_SPREAD(0.03), so no flooring: discount
    # rate stays 0.10. next_year_dps = 2.42*1.06 = 2.5652.
    value = engine.calculate_intrinsic_value()
    assert value == pytest.approx(2.5652 / 0.04)


def test_intrinsic_value_is_none_for_a_non_dividend_payer():
    df = _df(revenue=[100, 110])
    engine = DDMEngine(df, cost_of_equity=0.10)
    assert engine.calculate_intrinsic_value() is None


def test_intrinsic_value_is_none_when_cost_of_equity_is_unavailable():
    df = _df(
        dividends_paid=[100, 110, 121],
        shares_outstanding=[50, 50, 50],
        net_income=[500, 550, 600],
    )
    engine = DDMEngine(df, cost_of_equity=None)
    assert engine.calculate_intrinsic_value() is None


def test_intrinsic_value_never_divides_by_a_non_positive_spread():
    # cost_of_equity far below the dividend growth rate -- the floor
    # must still guarantee a positive, sane denominator.
    df = _df(
        dividends_paid=[100, 121],
        shares_outstanding=[50, 50],
        net_income=[500, 600],
    )
    engine = DDMEngine(df, cost_of_equity=0.01)
    value = engine.calculate_intrinsic_value()
    assert value is not None
    assert value > 0
