"""
pit_composite.py -- RESEARCH SANDBOX (not production code; nothing in app/ or scripts/ is modified).

P2b of SPRINT_TRACKER.md: run the UNCHANGED production scorer
(scripts/phase2_backtest._score_ticker_at_date -> DCF + relative valuation -> composite_score)
on EDGAR as-filed statements instead of yfinance's restated, ~5-year statements, so the
composite's IC can be measured over 2012-2025 instead of nine quarters.

The adapter's only job is to hand the production code the same table shapes yfinance would
(rows = line-item labels, columns = fiscal-year-end dates), built from facts that had been
filed on or before the as-of date.

Run:
    python research/pit_composite.py smoke            # 5 tickers x 2 dates, timing + sanity
    python research/pit_composite.py run              # everything pre-registered
"""

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEV_END = pd.Timestamp("2025-06-30")   # later dates are the sealed holdout
FIRST_DATE = pd.Timestamp("2012-06-29")
STATEMENT_YEARS = 4                     # yfinance hands production 4 fiscal years; the DCF growth estimate depends on the window

INCOME = {"Total Revenue": "revenue", "Operating Income": "operating_income", "Net Income": "net_income",
          "Tax Provision": "income_tax", "Pretax Income": "pretax_income", "Interest Expense": "interest_expense"}
CASHFLOW = {"Operating Cash Flow": "cfo", "Capital Expenditure": "capex",
            "Depreciation And Amortization": "depreciation", "Cash Dividends Paid": "dividends_paid"}
BALANCE = {"Current Assets": "current_assets", "Current Liabilities": "current_liabilities",
           "Cash And Cash Equivalents": "cash", "Total Assets": "total_assets",
           "Stockholders Equity": "equity", "Retained Earnings": "retained_earnings"}
NEEDED = set(INCOME.values()) | set(CASHFLOW.values()) | set(BALANCE.values()) | {
    "long_term_debt", "debt_current", "shares_outstanding", "diluted_shares"}


# ----------------------------------------------------------------------------- data prep (parent process)

def prepare():
    """Per-ticker slices of the EDGAR annual facts + split series + price series."""
    from research.pit_panel import annual_rows, load_facts

    facts = load_facts()
    ann = annual_rows(facts[facts["logical"].isin(NEEDED)])
    # best tag first, then latest filing first -> first row per (item, fiscal year) is the version known
    ann = ann.sort_values(["cik", "logical", "end", "tag_rank", "filed"], ascending=[True, True, True, True, False])
    mapping = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str})
    dual = mapping["cik"].duplicated(keep=False)
    mapping = mapping[~dual].drop_duplicates("ticker")        # dual-class share counts are ambiguous -> excluded
    by_cik = {c: g for c, g in ann.groupby("cik")}
    splits = pd.read_parquet(DATA / "splits.parquet")
    splits = {t: g.set_index("date")["ratio"].sort_index() for t, g in splits.groupby("ticker")}
    close = pd.read_parquet(DATA / "prices_close_splitadj.parquet")
    volume = pd.read_parquet(DATA / "prices_volume.parquet")
    jobs = []
    for _, row in mapping.iterrows():
        t, c = row["ticker"], row["cik"]
        if c in by_cik and t in close.columns:
            jobs.append((t, row["category"], by_cik[c], splits.get(t, pd.Series(dtype=float, index=pd.DatetimeIndex([]))),
                         pd.DataFrame({"Close": close[t], "Volume": volume[t]}).dropna(subset=["Close"])))
    return jobs


def market_series():
    import yfinance as yf

    out = {}
    for sym in ("^GSPC", "^TNX"):
        h = yf.Ticker(sym).history(start="2007-01-01", auto_adjust=False)
        h.index = h.index.tz_localize(None).normalize()
        out[sym] = h
    return out


# ----------------------------------------------------------------------------- adapter (worker side)

def _split_product(splits, after, upto=None):
    s = splits[splits.index > after]
    if upto is not None:
        s = s[s.index <= upto]
    return float(np.prod(s.values)) if len(s) else 1.0


def build_raw(ann, splits, prices, as_of):
    """The raw_data dict production expects, using only facts filed on or before `as_of`."""
    known = ann[ann["filed"] <= as_of]
    if known.empty:
        return None
    best = known.drop_duplicates(["logical", "end"], keep="first")
    # shares_outstanding rows are dated by the cover page, not the fiscal year-end: handled separately below
    wide = best[best["logical"] != "shares_outstanding"].pivot(index="end", columns="logical", values="val")
    ends = sorted(wide.index)[-STATEMENT_YEARS:]
    wide = wide.loc[ends]

    # Shares: cover-page count in the filing that FIRST reported each fiscal year, restated to today-at-as_of split basis.
    first_filing = (known[known["logical"].isin(["net_income", "revenue"])]
                    .sort_values("filed").drop_duplicates("end", keep="first").set_index("end"))
    shares_rows = known[known["logical"] == "shares_outstanding"].drop_duplicates("accn").set_index("accn")
    diluted = best[best["logical"] == "diluted_shares"].set_index("end")["val"]
    shares = {}
    for e in ends:
        val, filed = np.nan, None
        if e in first_filing.index:
            acc, filed = first_filing.loc[e, "accn"], first_filing.loc[e, "filed"]
            if isinstance(acc, pd.Series):
                acc, filed = acc.iloc[0], filed.iloc[0]
            if acc in shares_rows.index:
                val = shares_rows.loc[acc, "val"]
        if (np.isnan(val) or val <= 0) and e in diluted.index:
            val, filed = diluted[e], first_filing.loc[e, "filed"] if e in first_filing.index else e
            if isinstance(filed, pd.Series):
                filed = filed.iloc[0]
        if filed is not None and not np.isnan(val) and val > 0:
            shares[e] = val * _split_product(splits, filed, as_of)

    def frame(mapping):
        df = pd.DataFrame({label: wide[col] for label, col in mapping.items() if col in wide.columns}).T
        df.columns = pd.to_datetime(df.columns)
        return df

    income, cashflow, balance = frame(INCOME), frame(CASHFLOW), frame(BALANCE)
    # yfinance's "EBIT" row (production's first-choice alias) is pretax income + interest expense; many filers
    # (IBM, ROST...) never tag OperatingIncomeLoss, which would make the DCF unavailable for them.
    if income.empty or cashflow.empty or balance.empty:
        return None
    debt = wide.get("long_term_debt", pd.Series(0.0, index=wide.index)).fillna(0) +         wide.get("debt_current", pd.Series(0.0, index=wide.index)).fillna(0)
    # A company with no debt reports no interest expense; that is a true zero, not missing data.
    interest = wide.get("interest_expense", pd.Series(np.nan, index=wide.index))
    interest = interest.where(~(interest.isna() & (debt <= 0)), 0.0)
    income.loc["Interest Expense"] = interest.rename(index=pd.Timestamp)
    if "pretax_income" in wide.columns:
        ebit = wide["pretax_income"] + interest.fillna(0)
        if "operating_income" in wide.columns:
            ebit = ebit.fillna(wide["operating_income"])
        income.loc["EBIT"] = ebit.rename(index=pd.Timestamp)
    if "Capital Expenditure" in cashflow.index:
        cashflow.loc["Capital Expenditure"] = -cashflow.loc["Capital Expenditure"].abs()    # yfinance: outflow is negative
    balance.loc["Total Debt"] = debt.rename(index=pd.Timestamp)
    balance.loc["Ordinary Shares Number"] = pd.Series(shares).rename(index=pd.Timestamp)
    if income.empty or balance.empty or cashflow.empty:
        return None

    history = prices[prices.index <= as_of].copy()
    history["Close"] = history["Close"] * _split_product(splits, as_of)    # price on the same split basis as the shares
    for col in ("Open", "High", "Low"):
        history[col] = history["Close"]
    return {"price_history": history, "income": income, "balance": balance, "cashflow": cashflow}


_STATE = {}


def _init_worker(market):
    import app.tools.valuation_tool as vt

    def local_benchmark(ticker, period="5y"):          # no network inside workers
        return market.get(ticker)

    vt.get_benchmark_history = local_benchmark
    _STATE["market"] = market


def score_ticker(job_and_dates):
    from scripts.phase2_backtest import _score_ticker_at_date

    (ticker, category, ann, splits, prices), dates = job_and_dates
    market = _STATE["market"]
    rows = []
    for as_of in dates:
        if prices.index.min() > as_of - pd.Timedelta(days=300):
            continue                                        # need ~1y of trailing prices for beta
        try:
            raw = build_raw(ann, splits, prices, as_of)
            if raw is None:
                continue
            res = _score_ticker_at_date(ticker, category, raw, as_of, as_of, market["^GSPC"], market["^TNX"])
        except Exception as exc:                             # one bad ticker-date must not stop the run
            rows.append({"ticker": ticker, "date": as_of, "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
            continue
        rows.append({"ticker": ticker, "date": as_of, "composite_score": res["composite_score"],
                     "dcf_score": res["dcf_score"], "relative_score": res["relative_score"],
                     "recommendation": res["recommendation"], "upside_pct": res["upside_pct"], "error": None})
    return rows


def quarter_ends():
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    idx = adj.index.to_series()
    qe = list(idx.groupby(idx.index.to_period("Q")).max())
    return [d for d in qe if d >= FIRST_DATE]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("smoke", "run"):
        print(__doc__)
        return
    qe = quarter_ends()
    dates = [d for d in qe if d <= DEV_END]
    assert all(d <= DEV_END for d in dates)
    jobs = prepare()
    market = market_series()
    if mode == "smoke":
        sel = [j for j in jobs if j[0] in ("AAPL", "HD", "CAT", "WMS", "KNX")]
        dates = [d for d in dates if d in (pd.Timestamp("2014-06-30"), pd.Timestamp("2019-12-31"))]
        jobs = sel
    t0 = time.time()
    with Pool(min(16, len(jobs)), initializer=_init_worker, initargs=(market,)) as pool:
        results = []
        for i, chunk in enumerate(pool.imap_unordered(score_ticker, [(j, dates) for j in jobs], chunksize=1), 1):
            results.append(chunk)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} tickers done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    rows = [r for chunk in results for r in chunk]
    out = pd.DataFrame(rows)
    path = DATA / ("p2b_smoke.csv" if mode == "smoke" else "p2b_scores.csv")
    out.to_csv(path, index=False)
    ok = out[out["error"].isna()] if "error" in out else out
    print(f"{len(jobs)} tickers x {len(dates)} dates in {time.time() - t0:.0f}s -> {len(ok)} scored, {len(out) - len(ok)} errors")
    if "error" in out:
        print(out["error"].value_counts().head(8).to_string())
    print(ok.head(12).to_string())


if __name__ == "__main__":
    main()
