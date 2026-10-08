"""
edgar_pit.py -- RESEARCH SANDBOX (not production code; nothing in app/ imports this).

P1 of SPRINT_TRACKER.md: pull SEC EDGAR XBRL "company facts" for the FinSight
universe and keep a compact set of core line items WITH their original filing
dates, so any later analysis can ask "what did an investor actually know on
date T?" instead of using yfinance's restated, 5-year-limited statements.

Output (gitignored): research/data/edgar_facts.parquet, one row per
(company, line item, fiscal period, filing). The same fiscal period appears
once per filing that reported it, which is exactly what makes point-in-time
queries possible (take the latest `filed` <= T).

Run: python research/edgar_pit.py [--limit N]
Resumable: finished companies are cached as shards and skipped on re-run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.sec_edgar_client import HEADERS, SECEdgarClient  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SHARDS = DATA / "edgar_shards_v4"  # v4 adds CFO/capex/interest/D&A alternative tags (found by sampling companies whose inputs were missing); v3 added debt_current, retained_earnings, extra interest/revenue tags (needed to feed the production valuation model)
UNIVERSE_PATH = ROOT.parent / "scripts" / "ticker_universe.json"

REQUEST_PAUSE_SECONDS = 0.12  # SEC's published cap is 10 requests/second
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

# Logical line item -> XBRL tags to try. The SAME item is reported under
# different tags over the years (e.g. revenue: SalesRevenueNet before 2018,
# RevenueFromContractWithCustomer... after), so every tag is collected and
# a later step picks the first tag that has data for a given period.
CONCEPTS = {
    "revenue": [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense", "SalesRevenueServicesNet",
        "RealEstateRevenueNet", "OperatingLeasesIncomeStatementLeaseRevenue",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "income_tax": ["IncomeTaxExpenseBenefit"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense",
                         "InterestPaidNet", "InterestPaid"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets", "PaymentsForCapitalImprovements",
              "PaymentsToAcquireOtherPropertyPlantAndEquipment", "PaymentsToAcquireRealEstate"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation", "DepreciationAmortizationAndAccretionNet"],
    "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "total_assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_current": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}
TAXONOMIES = ("us-gaap", "dei")
COLUMNS = ["cik", "logical", "tag", "unit", "start", "end", "val", "fy", "fp", "form", "filed", "accn"]


def _get_json(session, url):
    """GET with backoff. Returns (json, status): 404 -> (None, 404), exhausted retries -> (None, last status)."""
    status = None
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=60)
            status = resp.status_code
            if resp.status_code == 200:
                return resp.json(), 200
            if resp.status_code == 404:
                return None, 404
        except requests.RequestException:
            status = "network"
        time.sleep(2 ** attempt)
    return None, status


def extract_rows(facts_json, cik):
    rows = []
    facts = facts_json.get("facts", {})
    for logical, tags in CONCEPTS.items():
        for tag in tags:
            for taxonomy in TAXONOMIES:
                node = facts.get(taxonomy, {}).get(tag)
                if not node:
                    continue
                for unit, items in node.get("units", {}).items():
                    for item in items:
                        if item.get("form") not in FORMS:
                            continue
                        rows.append((
                            cik, logical, tag, unit, item.get("start"), item["end"], item["val"],
                            item.get("fy"), item.get("fp"), item["form"], item["filed"], item.get("accn"),
                        ))
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N companies (smoke test)")
    args = parser.parse_args()

    SHARDS.mkdir(parents=True, exist_ok=True)
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        universe = json.load(f)

    client = SECEdgarClient()
    ticker_cik = {}
    for ticker in universe:
        cik = client.get_cik(ticker)
        if cik:
            ticker_cik[ticker] = cik
    unmapped = [t for t in universe if t not in ticker_cik]
    print(f"universe {len(universe)} tickers -> {len(ticker_cik)} mapped to a CIK, {len(unmapped)} unmapped", file=sys.stderr)

    pd.DataFrame(
        [{"ticker": t, "cik": c, "category": universe[t]} for t, c in ticker_cik.items()]
    ).to_csv(DATA / "ticker_cik.csv", index=False)

    ciks = list(dict.fromkeys(ticker_cik.values()))  # dual-class tickers share one CIK
    if args.limit:
        ciks = ciks[: args.limit]

    session = requests.Session()
    session.headers.update(HEADERS)
    missing = []
    for i, cik in enumerate(ciks, 1):
        shard = SHARDS / f"{cik}.parquet"
        if shard.exists():
            continue
        data, status = _get_json(session, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        if data is None:
            missing.append((cik, status))
        else:
            extract_rows(data, cik).to_parquet(shard, index=False)
        time.sleep(REQUEST_PAUSE_SECONDS)
        if i % 25 == 0 or i == len(ciks):
            print(f"  {i}/{len(ciks)} companies processed ({len(missing)} without facts)", file=sys.stderr)

    shards = sorted(SHARDS.glob("*.parquet"))
    frames = [pd.read_parquet(s) for s in shards]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    combined.to_parquet(DATA / "edgar_facts.parquet", index=False)
    pd.DataFrame(missing, columns=["cik", "status"]).to_csv(DATA / "edgar_missing.csv", index=False)
    print(f"\nwrote {len(combined):,} fact rows for {combined['cik'].nunique()} companies; "
          f"{len(missing)} companies had no XBRL facts (foreign filers, trusts, etc.)", file=sys.stderr)


if __name__ == "__main__":
    main()
