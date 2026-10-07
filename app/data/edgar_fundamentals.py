"""
edgar_fundamentals.py

Annual income statement / balance sheet / cash flow built from SEC EDGAR's
XBRL "company facts" API, shaped exactly like yfinance's `.financials` /
`.balance_sheet` / `.cashflow` DataFrames (rows = line-item labels, columns =
fiscal-year-end dates) so the existing FinancialStatementNormaliser consumes
them unchanged.

Why this exists: yfinance is this project's single point of failure for
fundamentals (it serves only ~4-5 restated years and throttles datacenter
IPs). EDGAR is free, unthrottled at the volumes this app needs, goes back 15+
years, and reports the numbers as originally filed. It is used as a FALLBACK
(FUNDAMENTALS_SOURCE=auto, the default: yfinance first, EDGAR only when
yfinance's statements fail or come back empty) or can be forced with
FUNDAMENTALS_SOURCE=edgar; FUNDAMENTALS_SOURCE=yfinance disables it.

Known, measured limits (see EVALUATION.md, "Longer-history re-test"): the
EDGAR-fed pipeline reproduces yfinance-fed composite scores with a 0.83 rank
correlation, not 1.0 -- yfinance's "Total Debt" includes lease obligations,
EDGAR's now includes lease liabilities (OperatingLeaseLiability, taken as the first tag that exists, so a company reporting both operating and finance leases is slightly under-counted), and share counts come from the 10-K cover page. Only
US filers with a CIK are covered; everything else returns None and the caller
keeps its existing behaviour.
"""

import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from app.core.paths import DATA_DIR
from app.core.retry import retry_on_transient_error
from app.data.sec_edgar_client import HEADERS, SECEdgarClient

CACHE_DIR = DATA_DIR / "edgar_facts_cache"
CACHE_TTL_SECONDS = 24 * 3600
ANNUAL_FORMS = {"10-K", "10-K/A"}
ANNUAL_DAYS = (340, 390)
MAX_YEARS = 4  # what yfinance hands the model; the DCF's growth estimate depends on the window

# Logical line item -> XBRL tags in order of preference (the same item is
# filed under different tags over the years).
CONCEPTS: Dict[str, List[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
                "RevenuesNetOfInterestExpense", "SalesRevenueServicesNet", "RealEstateRevenueNet",
                "OperatingLeasesIncomeStatementLeaseRevenue"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense",
                         "InterestPaidNet", "InterestPaid"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment", "PaymentsToAcquireRealEstate"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation",
                     "DepreciationAmortizationAndAccretionNet"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "lease_liabilities": ["OperatingLeaseLiability", "FinanceLeaseLiability"],  # yfinance Total Debt includes lease obligations
    "debt_current": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],   # dei: 10-K cover page, dated by filing not by fiscal year
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}
INSTANT = {"total_assets", "current_assets", "current_liabilities", "long_term_debt", "debt_current", "lease_liabilities", "cash", "equity",
           "retained_earnings", "shares_outstanding"}
TAXONOMIES = ("us-gaap", "dei")

INCOME_ROWS = {"Total Revenue": "revenue", "Operating Income": "operating_income", "Net Income": "net_income",
               "Tax Provision": "income_tax", "Pretax Income": "pretax_income"}
CASHFLOW_ROWS = {"Operating Cash Flow": "cfo", "Depreciation And Amortization": "depreciation",
                 "Cash Dividends Paid": "dividends_paid"}
BALANCE_ROWS = {"Current Assets": "current_assets", "Current Liabilities": "current_liabilities",
                "Cash And Cash Equivalents": "cash", "Total Assets": "total_assets", "Stockholders Equity": "equity",
                "Retained Earnings": "retained_earnings"}


def fetch_company_facts(cik: str) -> Optional[dict]:
    """The raw XBRL company-facts JSON, cached on disk for a day."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"CIK{cik}.json"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL_SECONDS:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    def _do():
        resp = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp

    resp = retry_on_transient_error(_do)
    if resp is None:
        return None
    data = resp.json()
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # a cache write failure must never fail the fetch
    return data


def _annual_rows(facts_json: dict) -> pd.DataFrame:
    rows = []
    facts = facts_json.get("facts", {})
    for logical, tags in CONCEPTS.items():
        for rank, tag in enumerate(tags):
            for taxonomy in TAXONOMIES:
                node = facts.get(taxonomy, {}).get(tag)
                if not node:
                    continue
                for items in node.get("units", {}).values():
                    for it in items:
                        if it.get("form") not in ANNUAL_FORMS:
                            continue
                        rows.append((logical, rank, it.get("start"), it["end"], it["val"], it["filed"], it.get("accn")))
    df = pd.DataFrame(rows, columns=["logical", "rank", "start", "end", "val", "filed", "accn"])
    if df.empty:
        return df
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    days = (df["end"] - df["start"]).dt.days
    return df[df["logical"].isin(INSTANT) | days.between(*ANNUAL_DAYS)]


def build_statements(facts_json: dict, splits: Optional[pd.Series] = None,
                     max_years: int = MAX_YEARS) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """(income, balance, cashflow) in yfinance shape from a company-facts JSON, or None if too little data."""
    df = _annual_rows(facts_json)
    if df.empty:
        return None
    # Best tag first, then the latest filing (a later 10-K restates the prior year: take the most recent figure).
    df = df.sort_values(["logical", "end", "rank", "filed"], ascending=[True, True, True, False])
    best = df.drop_duplicates(["logical", "end"], keep="first")
    wide = best[best["logical"] != "shares_outstanding"].pivot(index="end", columns="logical", values="val")
    if "net_income" not in wide.columns and "revenue" not in wide.columns:
        return None
    wide = wide[wide[[c for c in ("revenue", "net_income") if c in wide.columns]].notna().any(axis=1)]
    ends = sorted(wide.index)[-max_years:]
    if len(ends) < 2:
        return None
    wide = wide.loc[ends]

    # Shares: the cover-page count in the filing that FIRST reported each fiscal year, restated for later splits.
    first = (df[df["logical"].isin(["net_income", "revenue"])].sort_values("filed")
             .drop_duplicates("end", keep="first").set_index("end"))
    cover = df[df["logical"] == "shares_outstanding"].drop_duplicates("accn").set_index("accn")["val"]
    diluted = best[best["logical"] == "diluted_shares"].set_index("end")["val"]
    shares = {}
    for e in ends:
        val, filed = np.nan, first["filed"].get(e)
        acc = first["accn"].get(e)
        if acc in cover.index:
            val = float(cover[acc])
        if (np.isnan(val) or val <= 0) and e in diluted.index:
            val = float(diluted[e])
        if filed is not None and not np.isnan(val) and val > 0:
            if splits is not None and len(splits):
                later = splits[splits.index > filed]
                val *= float(np.prod(later.values)) if len(later) else 1.0
            shares[pd.Timestamp(e)] = val

    def frame(mapping: Dict[str, str]) -> pd.DataFrame:
        cols = {label: wide[col] for label, col in mapping.items() if col in wide.columns}
        out = pd.DataFrame(cols).T
        out.columns = pd.to_datetime(out.columns)
        return out

    income, cashflow, balance = frame(INCOME_ROWS), frame(CASHFLOW_ROWS), frame(BALANCE_ROWS)
    if income.empty or cashflow.empty or balance.empty:
        return None
    zero = pd.Series(0.0, index=wide.index)
    debt = (wide.get("long_term_debt", zero).fillna(0) + wide.get("debt_current", zero).fillna(0)
            + wide.get("lease_liabilities", zero).fillna(0))
    # A company with no debt reports no interest expense: a true zero, not missing data.
    interest = wide.get("interest_expense", pd.Series(np.nan, index=wide.index))
    interest = interest.where(~(interest.isna() & (debt <= 0)), 0.0)
    income.loc["Interest Expense"] = interest.rename(index=pd.Timestamp)
    if "pretax_income" in wide.columns:
        # yfinance's "EBIT" is pretax income + interest expense; many filers never tag OperatingIncomeLoss.
        ebit = wide["pretax_income"] + interest.fillna(0)
        if "operating_income" in wide.columns:
            ebit = ebit.fillna(wide["operating_income"])
        income.loc["EBIT"] = ebit.rename(index=pd.Timestamp)
    if "capex" in wide.columns:
        cashflow.loc["Capital Expenditure"] = -wide["capex"].abs().rename(index=pd.Timestamp)  # yfinance: outflow negative
    balance.loc["Total Debt"] = debt.rename(index=pd.Timestamp)
    balance.loc["Ordinary Shares Number"] = pd.Series(shares)
    return income, balance, cashflow


def get_statements(ticker: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Live EDGAR statements for a US ticker, or None if the ticker has no CIK / no usable XBRL facts.
    Never raises: a fallback source must not turn a recoverable yfinance failure into a crash."""
    try:
        cik = SECEdgarClient().get_cik(ticker)
        if not cik:
            return None
        facts = fetch_company_facts(cik)
        if not facts:
            return None
        splits = None
        try:  # best-effort: only used to restate older share counts onto today's split basis
            import yfinance as yf
            s = yf.Ticker(ticker).splits
            if s is not None and len(s):
                s.index = s.index.tz_localize(None).normalize()
                splits = s[s > 0]
        except Exception:
            splits = None
        return build_statements(facts, splits)
    except Exception:
        return None
