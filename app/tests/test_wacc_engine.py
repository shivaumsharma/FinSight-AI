"""
Unit tests for WACCEngine (app/valuation/wacc_engine.py).

No network, no models -- pure pandas/numpy math against small
hand-built fixtures, so these run in milliseconds and belong in CI.
"""

import pandas as pd
import pytest

from app.valuation.wacc_engine import WACCEngine


def _financial_df(**overrides):
    base = {
        "interest_expense": [-50, -60],
        "total_debt": [1000, 1200],
        "tax_expense": [100, 120],
        "pretax_income": [500, 600],
    }
    base.update(overrides)
    return pd.DataFrame(base, index=pd.to_datetime(["2023-12-31", "2024-12-31"]))


def test_cost_of_equity_is_capm():
    engine = WACCEngine(
        financial_df=_financial_df(), market_cap=10_000, beta=1.5,
        risk_free_rate=0.04, market_risk_premium=0.06,
    )
    # CAPM: rf + beta * ERP
    assert engine.calculate_cost_of_equity() == pytest.approx(0.04 + 1.5 * 0.06)


def test_cost_of_debt_averages_interest_over_debt():
    engine = WACCEngine(financial_df=_financial_df(), market_cap=10_000, beta=1.0)
    # |interest| / debt per year: 50/1000=0.05, 60/1200=0.05 -> mean 0.05
    assert engine.calculate_cost_of_debt() == pytest.approx(0.05)


def test_cost_of_debt_excludes_zero_debt_years():
    df = _financial_df(total_debt=[0, 1200], interest_expense=[0, -60])
    engine = WACCEngine(financial_df=df, market_cap=10_000, beta=1.0)
    # Only the second year (debt > 0) should count: 60/1200 = 0.05
    assert engine.calculate_cost_of_debt() == pytest.approx(0.05)


def test_tax_rate_is_clipped_to_50_percent():
    # A tax_expense/pretax_income ratio above 0.5 (e.g. a one-off
    # deferred-tax charge) must not blow up the WACC's after-tax cost
    # of debt term -- clip(0, 0.5) exists specifically for this.
    df = _financial_df(tax_expense=[400, 120], pretax_income=[500, 600])
    engine = WACCEngine(financial_df=df, market_cap=10_000, beta=1.0)
    tax_rate = engine.calculate_tax_rate()
    assert tax_rate <= 0.5


def test_tax_rate_ignores_non_positive_pretax_income():
    # A loss year (pretax_income <= 0) produces a meaningless tax
    # "rate" and must be excluded from the average, not included as-is.
    df = _financial_df(tax_expense=[10, 120], pretax_income=[-500, 600])
    engine = WACCEngine(financial_df=df, market_cap=10_000, beta=1.0)
    assert engine.calculate_tax_rate() == pytest.approx(120 / 600)


def test_wacc_blends_equity_and_debt_by_market_weight():
    market_cap = 9_000
    df = _financial_df(total_debt=[1000, 1000], interest_expense=[-50, -50],
                        tax_expense=[100, 100], pretax_income=[500, 500])
    engine = WACCEngine(financial_df=df, market_cap=market_cap, beta=1.0,
                         risk_free_rate=0.04, market_risk_premium=0.06)

    cost_of_equity = engine.calculate_cost_of_equity()  # 0.10
    cost_of_debt = engine.calculate_cost_of_debt()       # 0.05
    tax_rate = engine.calculate_tax_rate()               # 0.20
    total_value = market_cap + 1000

    expected = (market_cap / total_value) * cost_of_equity + \
               (1000 / total_value) * cost_of_debt * (1 - tax_rate)

    assert engine.calculate_wacc() == pytest.approx(expected)
