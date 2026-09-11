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


def test_revenue_cagr_returns_none_for_a_single_data_point():
    # n = len(revenue_series) - 1 would be 0, making 1/n a
    # ZeroDivisionError -- must degrade to None instead of raising.
    df = _df(revenue=[100])
    engine = FCFFEngine(df)
    assert engine.calculate_revenue_cagr() is None


def test_revenue_cagr_returns_none_for_a_non_positive_beginning_value():
    # A zero-or-negative starting value makes (ending/beginning)**(1/n)
    # either undefined or a meaningless complex/negative-base result.
    df = _df(revenue=[0, 100])
    engine = FCFFEngine(df)
    assert engine.calculate_revenue_cagr() is None


def test_forecast_revenue_returns_none_when_cagr_is_uncomputable():
    # forecast_revenue must propagate calculate_revenue_cagr()'s None
    # rather than assume a numeric growth rate exists.
    df = _df(revenue=[100])
    engine = FCFFEngine(df)
    assert engine.forecast_revenue() is None


def test_forecast_revenue_compounds_at_cagr():
    df = _df(revenue=[100, 110])
    engine = FCFFEngine(df)
    forecast = engine.forecast_revenue(forecast_years=2)
    cagr = engine.calculate_revenue_cagr()
    assert forecast["forecast_revenue"].iloc[0] == pytest.approx(110 * (1 + cagr))
    assert forecast["forecast_revenue"].iloc[1] == pytest.approx(110 * (1 + cagr) ** 2)


def test_calculate_tax_rate_excludes_a_break_even_or_loss_year():
    # A pretax_income<=0 year has no meaningful effective tax rate --
    # tax_expense/pretax_income for such a year isn't real, and letting
    # it through used to clip to a fabricated 50% ceiling (this was the
    # bug wacc_engine.py's own calculate_tax_rate already guarded
    # against but fcff_engine.py's duplicate copy didn't). Shared
    # app.valuation.tax_rate.calculate_tax_rate now excludes it
    # entirely for both engines.
    df = _df(tax_expense=[-10, 40], pretax_income=[0, 200])
    engine = FCFFEngine(df)
    tax_rate = engine.calculate_tax_rate()
    assert len(tax_rate) == 1
    assert tax_rate.iloc[0] == pytest.approx(40 / 200)


def test_calculate_nopat_excludes_a_break_even_year_instead_of_a_fabricated_tax_rate():
    df = _df(ebit=[30, 250], tax_expense=[-10, 40], pretax_income=[0, 200])
    engine = FCFFEngine(df)
    nopat = engine.calculate_nopat()
    assert len(nopat) == 1
    assert nopat.iloc[0] == pytest.approx(250 * (1 - 40 / 200))


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


def test_forecast_fcff_clips_an_extreme_override_growth_rate_to_the_ceiling():
    # A caller-supplied 80% initial growth rate (well above
    # MAX_INITIAL_GROWTH_RATE) must be clipped before Stage 1 compounds
    # it, not applied as-is -- the ceiling is enforced unconditionally,
    # not only on the freshly-computed (non-override) path.
    df = _df(revenue=[100, 130], net_income=[10, 12], total_equity=[50, 90])  # ROE 90/... high quality tier
    engine = FCFFEngine(df)
    forecast = engine.forecast_fcff(
        forecast_years=10, terminal_growth_rate=0.03,
        base_fcff_override=100.0, initial_growth_rate_override=0.80,
    )
    year1_growth = forecast["forecast_fcff"].iloc[0] / 100.0 - 1
    assert year1_growth == pytest.approx(FCFFEngine.MAX_INITIAL_GROWTH_RATE, abs=1e-6)
    assert year1_growth < 0.80


def test_forecast_fcff_clips_a_hypergrowth_companys_own_raw_revenue_cagr():
    # Reproduces the audit's concrete failure shape: revenue roughly
    # quadrupling over 4 fiscal years (~78% CAGR), a high-ROE company
    # (5-year Stage 1 hold) -- without a ceiling, base FCFF compounds
    # at ~78%/yr for 5 straight years before the fade even starts. Not
    # a hand-picked override this time -- the raw CAGR the engine
    # itself computes from revenue.
    df = _df(
        revenue=[50, 90, 160, 280, 500],
        net_income=[5, 9, 16, 28, 50], total_equity=[20, 25, 32, 40, 50],  # ROE 100% -> high-quality tier
        ebit=[8, 14, 25, 44, 79], depreciation=[1, 1, 2, 3, 5],
        capex=[2, 3, 5, 9, 16], current_assets=[10, 15, 22, 32, 46],
        current_liabilities=[5, 7, 10, 14, 20],
        tax_expense=[1, 2, 3, 6, 10], pretax_income=[8, 14, 25, 44, 79],
    )
    engine = FCFFEngine(df)
    raw_cagr = engine.calculate_revenue_cagr()
    assert raw_cagr > 0.30  # confirm this fixture actually reproduces an above-ceiling CAGR

    forecast = engine.forecast_fcff(forecast_years=10, terminal_growth_rate=0.03)
    base_fcff = engine.calculate_normalized_base_fcff()
    year1_growth = forecast["forecast_fcff"].iloc[0] / base_fcff - 1
    assert year1_growth == pytest.approx(FCFFEngine.MAX_INITIAL_GROWTH_RATE, abs=1e-6)
