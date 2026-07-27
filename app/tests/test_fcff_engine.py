"""
Unit tests for FCFFEngine (app/valuation/fcff_engine.py) -- revenue
CAGR, capex/NWC normalization, ROE-tiered growth-fade length, and the
None-vs-negative distinction in calculate_normalized_base_fcff (banks
with no EBIT/capex concept at all vs. a genuinely negative FCFF base).
"""

import pandas as pd
import pytest

from app.valuation.fcff_engine import FCFFEngine


def _df(**cols):
    n = len(next(iter(cols.values())))
    index = pd.to_datetime([f"20{20+i}-12-31" for i in range(n)])
    return pd.DataFrame(cols, index=index)


def test_revenue_cagr_over_multiple_years():
    df = _df(revenue=[100, 121])  # 2 years -> 1 period -> 21% CAGR
    engine = FCFFEngine(df)
    assert engine.calculate_revenue_cagr() == pytest.approx(0.21)


def test_forecast_revenue_compounds_at_cagr():
    df = _df(revenue=[100, 110])
    engine = FCFFEngine(df)
    forecast = engine.forecast_revenue(forecast_years=2)
    cagr = engine.calculate_revenue_cagr()
    assert forecast["forecast_revenue"].iloc[0] == pytest.approx(110 * (1 + cagr))
    assert forecast["forecast_revenue"].iloc[1] == pytest.approx(110 * (1 + cagr) ** 2)


def test_normalized_capex_uses_multi_year_average_ratio_not_latest_year():
    # Ratios: 10/100=0.10, 40/200=0.20 (an outlier capex spike) -> avg 0.15
    # applied to the CURRENT (latest) revenue of 200.
    df = _df(revenue=[100, 200], capex=[10, 40])
    engine = FCFFEngine(df)
    assert engine.calculate_normalized_capex() == pytest.approx(0.15 * 200)


def test_normalized_base_fcff_is_none_when_structurally_missing():
    # A bank-shaped statement: normaliser.normalise() always creates
    # every column (see financial_normalizer.py), but for a bank ebit/
    # capex/current-asset-split come back as all-NaN, not just absent
    # for one year -- that's the real shape this guards against.
    df = _df(
        revenue=[100, 110], net_income=[10, 11], cash_from_operations=[15, 16],
        ebit=[None, None], depreciation=[None, None], capex=[None, None],
        current_assets=[None, None], current_liabilities=[None, None],
        tax_expense=[1, 1], pretax_income=[5, 5],
    )
    engine = FCFFEngine(df)
    assert engine.calculate_normalized_base_fcff() is None


def test_normalized_base_fcff_can_be_negative_when_computable():
    # A company whose working capital is consuming cash faster than its
    # operations generate it (current_assets growing much faster than
    # current_liabilities, i.e. a large positive change in NWC) --
    # computable, just negative. Caller (ValuationPipeline) is
    # responsible for treating this as "skip DCF," not this method.
    df = _df(
        revenue=[100, 110], ebit=[5, 5], depreciation=[1, 1],
        capex=[1, 1], current_assets=[10, 100], current_liabilities=[10, 10],
        tax_expense=[1, 1], pretax_income=[5, 5],
    )
    engine = FCFFEngine(df)
    base = engine.calculate_normalized_base_fcff()
    assert base is not None
    assert base < 0


@pytest.mark.parametrize(
    "net_income,total_equity,expected_hold_years",
    [
        (30, 100, FCFFEngine.HOLD_YEARS_HIGH_QUALITY),   # ROE 30% -> high quality
        (20, 100, FCFFEngine.HOLD_YEARS_DEFAULT),        # ROE 20% -> mid
        (10, 100, FCFFEngine.HOLD_YEARS_LOW_QUALITY),    # ROE 10% -> low
        (150, 100, FCFFEngine.HOLD_YEARS_DEFAULT),       # ROE 150% -> implausible, treated as default
    ],
)
def test_quality_hold_years_is_roe_tiered(net_income, total_equity, expected_hold_years):
    df = _df(net_income=[net_income], total_equity=[total_equity])
    engine = FCFFEngine(df)
    assert engine._quality_hold_years() == expected_hold_years


def test_quality_hold_years_defaults_when_roe_unreadable():
    # Non-positive equity makes ROE a leverage artifact, not a quality
    # signal -- must fall back to the default tier, not error.
    df = _df(net_income=[10], total_equity=[-5])
    engine = FCFFEngine(df)
    assert engine._quality_hold_years() == FCFFEngine.HOLD_YEARS_DEFAULT


def test_forecast_fcff_reaches_terminal_growth_by_final_year():
    df = _df(revenue=[100, 130], net_income=[10, 12], total_equity=[50, 55])
    engine = FCFFEngine(df)
    forecast = engine.forecast_fcff(
        forecast_years=10, terminal_growth_rate=0.03,
        base_fcff_override=100.0, initial_growth_rate_override=0.30,
    )
    # Year-over-year growth of the LAST forecast step should have faded
    # down to (approximately) terminal growth, not still be near 30%.
    final_growth = forecast["forecast_fcff"].iloc[-1] / forecast["forecast_fcff"].iloc[-2] - 1
    assert final_growth == pytest.approx(0.03, abs=1e-6)
