"""
Unit tests for FinancialStatementNormaliser (app/data/financial_normalizer.py)
-- the layer that maps yfinance's inconsistent line-item naming onto
one canonical schema every valuation formula depends on.
"""

import pandas as pd

from app.data.financial_normalizer import FinancialStatementNormaliser


def _statement(index_labels, columns, data):
    """Mimics yfinance's raw shape: rows=line items, columns=fiscal
    year-end dates."""
    return pd.DataFrame(data, index=index_labels, columns=columns)


def test_extract_metric_uses_first_matching_alias():
    # metric_mappings.py lists "Total Revenue" before "Operating Revenue"
    # -- when a statement has both, the first alias found wins.
    income = _statement(
        ["Total Revenue", "Operating Revenue"],
        ["2023-12-31", "2024-12-31"],
        [[100, 110], [999, 999]],
    )
    normaliser = FinancialStatementNormaliser(income, pd.DataFrame(), pd.DataFrame())
    series = normaliser.extract_metric(income, ["Total Revenue", "Operating Revenue"])
    assert list(series) == [100, 110]


def test_extract_metric_falls_back_to_second_alias_when_first_is_absent():
    income = _statement(["Operating Revenue"], ["2023-12-31"], [[50]])
    normaliser = FinancialStatementNormaliser(income, pd.DataFrame(), pd.DataFrame())
    series = normaliser.extract_metric(income, ["Total Revenue", "Operating Revenue"])
    assert list(series) == [50]


def test_extract_metric_returns_empty_series_when_no_alias_matches():
    income = _statement(["Something Unrelated"], ["2023-12-31"], [[1]])
    normaliser = FinancialStatementNormaliser(income, pd.DataFrame(), pd.DataFrame())
    series = normaliser.extract_metric(income, ["Total Revenue"])
    assert series.empty


def test_align_and_clean_takes_absolute_value_of_capex():
    # yfinance reports capex as a negative outflow; every valuation
    # formula (FCFFEngine) expects a positive magnitude.
    df = pd.DataFrame({"capex": [-50, -60]}, index=pd.to_datetime(["2023-12-31", "2024-12-31"]))
    normaliser = FinancialStatementNormaliser(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    cleaned = normaliser.align_and_clean(df)
    assert list(cleaned["capex"]) == [50, 60]


def test_align_and_clean_sorts_by_date_ascending():
    df = pd.DataFrame({"revenue": [200, 100]}, index=pd.to_datetime(["2024-12-31", "2023-12-31"]))
    normaliser = FinancialStatementNormaliser(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    cleaned = normaliser.align_and_clean(df)
    assert list(cleaned["revenue"]) == [100, 200]


def test_align_and_clean_drops_rows_that_are_entirely_nan():
    df = pd.DataFrame(
        {"revenue": [100, None]}, index=pd.to_datetime(["2023-12-31", "2024-12-31"]),
    )
    normaliser = FinancialStatementNormaliser(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    cleaned = normaliser.align_and_clean(df)
    assert len(cleaned) == 1


def test_extract_metric_finds_total_assets():
    # Confirmed against a live yfinance fetch (AAPL and RELIANCE.NS both
    # use this exact label) before adding it to METRIC_MAPPINGS.
    balance_sheet = _statement(["Total Assets"], ["2023-12-31"], [[1000]])
    normaliser = FinancialStatementNormaliser(pd.DataFrame(), balance_sheet, pd.DataFrame())
    series = normaliser.extract_metric(balance_sheet, ["Total Assets"])
    assert list(series) == [1000]


def test_extract_metric_falls_back_to_second_alias_for_retained_earnings():
    balance_sheet = _statement(["Retained Earnings Accumulated Deficit"], ["2023-12-31"], [[500]])
    normaliser = FinancialStatementNormaliser(pd.DataFrame(), balance_sheet, pd.DataFrame())
    series = normaliser.extract_metric(
        balance_sheet, ["Retained Earnings", "Retained Earnings Accumulated Deficit"]
    )
    assert list(series) == [500]


def test_normalise_includes_total_assets_and_retained_earnings_columns():
    income = _statement(["Total Revenue"], ["2023-12-31"], [[100]])
    balance_sheet = _statement(
        ["Total Assets", "Retained Earnings"], ["2023-12-31"], [[1000], [400]]
    )
    normaliser = FinancialStatementNormaliser(income, balance_sheet, pd.DataFrame())
    df = normaliser.normalise()
    assert df["total_assets"].iloc[0] == 1000
    assert df["retained_earnings"].iloc[0] == 400


def test_validate_financials_flags_negative_revenue():
    df = pd.DataFrame({"revenue": [100, -10]})
    normaliser = FinancialStatementNormaliser(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    warnings = normaliser.validate_financials(df)
    assert any("Negative or zero" in w for w in warnings)
