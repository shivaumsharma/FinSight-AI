"""Tests for app/data/edgar_fundamentals.py and the EDGAR fallback in MarketDataLoader.
All offline: the SEC company-facts JSON is synthesized."""

import pandas as pd
import pytest

from app.data import edgar_fundamentals as ef
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.data.market_data import MarketDataLoader


def _fact(end, val, filed, accn, start=None, form="10-K"):
    d = {"end": end, "val": val, "filed": filed, "accn": accn, "form": form, "fy": int(end[:4]), "fp": "FY"}
    if start:
        d["start"] = start
    return d


def _node(items):
    return {"units": {"USD": items}}


def _facts(years=(2019, 2020, 2021, 2022, 2023), with_operating_income=True, with_debt=True, restate_2022=False):
    """A company with one 10-K per year (filed the following March)."""
    flow = lambda base, step: [  # noqa: E731
        _fact(f"{y}-12-31", base + step * i, f"{y + 1}-03-01", f"acc-{y}", start=f"{y}-01-01") for i, y in enumerate(years)
    ]
    inst = lambda base, step: [  # noqa: E731
        _fact(f"{y}-12-31", base + step * i, f"{y + 1}-03-01", f"acc-{y}") for i, y in enumerate(years)
    ]
    gaap = {
        "Revenues": _node(flow(1000, 100)),
        "NetIncomeLoss": _node(flow(100, 10)),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": _node(flow(130, 13)),
        "IncomeTaxExpenseBenefit": _node(flow(30, 3)),
        "NetCashProvidedByUsedInOperatingActivities": _node(flow(150, 15)),
        "PaymentsToAcquirePropertyPlantAndEquipment": _node(flow(40, 4)),
        "DepreciationDepletionAndAmortization": _node(flow(25, 2)),
        "Assets": _node(inst(2000, 100)),
        "AssetsCurrent": _node(inst(600, 30)),
        "LiabilitiesCurrent": _node(inst(400, 20)),
        "CashAndCashEquivalentsAtCarryingValue": _node(inst(200, 10)),
        "StockholdersEquity": _node(inst(900, 50)),
    }
    if with_operating_income:
        gaap["OperatingIncomeLoss"] = _node(flow(140, 14))
    if with_debt:
        gaap["LongTermDebtNoncurrent"] = _node(inst(500, 0))
        gaap["DebtCurrent"] = _node(inst(50, 0))
        gaap["InterestExpense"] = _node(flow(20, 1))
    # noise that must be ignored: a 10-Q row and a quarter-length (90 day) flow in a 10-K
    gaap["Revenues"]["units"]["USD"].append(_fact("2023-03-31", 9999, "2023-05-01", "acc-q", start="2023-01-01", form="10-Q"))
    gaap["Revenues"]["units"]["USD"].append(_fact("2023-12-31", 8888, "2024-03-01", "acc-2023", start="2023-10-01"))
    if restate_2022:  # the 2023 10-K restates fiscal 2022 revenue
        gaap["Revenues"]["units"]["USD"].append(_fact("2022-12-31", 1234, "2024-03-01", "acc-2023", start="2022-01-01"))
    dei = {"EntityCommonStockSharesOutstanding": _node([
        _fact(f"{y + 1}-02-15", 100 + 10 * i, f"{y + 1}-03-01", f"acc-{y}") for i, y in enumerate(years)
    ])}
    return {"facts": {"us-gaap": gaap, "dei": dei}}


def test_shapes_and_yfinance_labels():
    income, balance, cashflow = ef.build_statements(_facts())
    assert {"Total Revenue", "Net Income", "Pretax Income", "Tax Provision", "Interest Expense", "EBIT"} <= set(income.index)
    assert {"Total Debt", "Ordinary Shares Number", "Total Assets", "Cash And Cash Equivalents"} <= set(balance.index)
    assert {"Operating Cash Flow", "Capital Expenditure"} <= set(cashflow.index)
    assert (cashflow.loc["Capital Expenditure"] < 0).all()  # yfinance convention: outflow is negative
    assert all(isinstance(c, pd.Timestamp) for c in income.columns)


def test_caps_at_four_most_recent_years():
    income, _, _ = ef.build_statements(_facts())
    assert list(income.columns.year) == [2020, 2021, 2022, 2023]


def test_ignores_quarterly_forms_and_short_duration_flows():
    income, _, _ = ef.build_statements(_facts())
    assert income.loc["Total Revenue", pd.Timestamp("2023-12-31")] == 1400  # not 9999 (10-Q) or 8888 (90-day)


def test_latest_filing_wins_for_restated_year():
    income, _, _ = ef.build_statements(_facts(restate_2022=True))
    assert income.loc["Total Revenue", pd.Timestamp("2022-12-31")] == 1234


def test_total_debt_is_noncurrent_plus_current():
    _, balance, _ = ef.build_statements(_facts())
    assert balance.loc["Total Debt", pd.Timestamp("2023-12-31")] == 550


def test_lease_liabilities_are_included_in_total_debt():
    facts = _facts()
    facts["facts"]["us-gaap"]["OperatingLeaseLiability"] = _node([
        _fact(f"{y}-12-31", 70, f"{y + 1}-03-01", f"acc-{y}") for y in (2019, 2020, 2021, 2022, 2023)])
    _, balance, _ = ef.build_statements(facts)
    assert balance.loc["Total Debt", pd.Timestamp("2023-12-31")] == 620  # 500 + 50 + 70


def test_ebit_is_pretax_plus_interest_even_without_operating_income():
    income, _, _ = ef.build_statements(_facts(with_operating_income=False))
    last = pd.Timestamp("2023-12-31")
    assert income.loc["EBIT", last] == income.loc["Pretax Income", last] + income.loc["Interest Expense", last]


def test_debt_free_company_gets_zero_interest_not_missing():
    income, balance, _ = ef.build_statements(_facts(with_debt=False))
    assert (balance.loc["Total Debt"] == 0).all()
    assert (income.loc["Interest Expense"] == 0).all()


def test_shares_come_from_the_filing_that_first_reported_the_year_and_are_split_restated():
    splits = pd.Series([2.0], index=pd.to_datetime(["2022-06-01"]))
    _, balance, _ = ef.build_statements(_facts(), splits)
    shares = balance.loc["Ordinary Shares Number"]
    # fiscal 2020 (filed 2021-03, cover 110) predates the split -> doubled; fiscal 2023 (filed 2024-03, cover 140) does not
    assert shares[pd.Timestamp("2020-12-31")] == 110 * 2
    assert shares[pd.Timestamp("2023-12-31")] == 140


def test_returns_none_with_fewer_than_two_fiscal_years():
    assert ef.build_statements(_facts(years=(2023,))) is None
    assert ef.build_statements({"facts": {"us-gaap": {}, "dei": {}}}) is None


def test_output_feeds_the_existing_normaliser_unchanged():
    income, balance, cashflow = ef.build_statements(_facts())
    df = FinancialStatementNormaliser(income, balance, cashflow).normalise()
    for col in ("revenue", "ebit", "net_income", "cash_from_operations", "capex", "total_debt", "shares_outstanding",
                "interest_expense", "depreciation"):
        assert df[col].notna().all(), col
    assert (df["capex"] > 0).all()  # normaliser takes abs()


def test_get_statements_never_raises(monkeypatch):
    monkeypatch.setattr(ef.SECEdgarClient, "get_cik", lambda self, t: (_ for _ in ()).throw(RuntimeError("boom")))
    assert ef.get_statements("AAPL") is None


# ---------------------------------------------------------------- loader fallback

class _BrokenYF:
    @property
    def financials(self):
        raise RuntimeError("throttled")

    balance_sheet = cashflow = financials


def _loader(monkeypatch, source):
    monkeypatch.setenv("FUNDAMENTALS_SOURCE", source)
    loader = MarketDataLoader("TEST")
    loader.stock = _BrokenYF()
    monkeypatch.setattr("app.data.market_data.cache_get", lambda key: None)
    monkeypatch.setattr("app.data.market_data.cache_set", lambda *a, **k: None)
    monkeypatch.setattr("app.data.market_data.retry_on_transient_error", lambda fn, *a, **k: fn())
    return loader


def test_auto_mode_falls_back_to_edgar_when_yfinance_fails(monkeypatch):
    stmts = ef.build_statements(_facts())
    monkeypatch.setattr(ef, "get_statements", lambda t: stmts)
    loader = _loader(monkeypatch, "auto")
    assert "Total Revenue" in loader.get_income_statement().index
    assert "Total Debt" in loader.get_balance_sheet().index
    assert "Operating Cash Flow" in loader.get_cash_flow().index


def test_auto_mode_reraises_original_error_when_edgar_has_nothing(monkeypatch):
    monkeypatch.setattr(ef, "get_statements", lambda t: None)
    with pytest.raises(RuntimeError, match="throttled"):
        _loader(monkeypatch, "auto").get_income_statement()


def test_yfinance_mode_never_uses_edgar(monkeypatch):
    called = []
    monkeypatch.setattr(ef, "get_statements", lambda t: called.append(t))
    with pytest.raises(RuntimeError, match="throttled"):
        _loader(monkeypatch, "yfinance").get_income_statement()
    assert called == []


def test_invalid_env_value_defaults_to_auto(monkeypatch):
    monkeypatch.setenv("FUNDAMENTALS_SOURCE", "nonsense")
    assert MarketDataLoader.fundamentals_source() == "auto"
