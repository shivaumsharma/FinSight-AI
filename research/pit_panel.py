"""
pit_panel.py -- RESEARCH SANDBOX (not production code).

P1 of SPRINT_TRACKER.md, step 2: answer "what did an investor know on date T?"
from the EDGAR facts built by edgar_pit.py, and measure how much yfinance's
restated numbers differ from what was originally reported.

Usage:
    python research/pit_panel.py validate     # restatement stats + yfinance cross-check
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.edgar_pit import CONCEPTS  # noqa: E402

ROOT = Path(__file__).resolve().parent
FACTS_PATH = ROOT / "data" / "edgar_facts.parquet"

# Balance-sheet style items are point-in-time ("instant") values with no start date;
# everything else is a flow over a period and must span about a year to be an annual figure.
INSTANT_ITEMS = {
    "total_assets", "current_assets", "current_liabilities", "long_term_debt",
    "cash", "equity", "shares_outstanding",
}
ANNUAL_FORMS = {"10-K", "10-K/A"}
ANNUAL_DAYS = (340, 390)
MAX_STALENESS_DAYS = 550  # latest fiscal year must end within ~18 months of T


def load_facts():
    df = pd.read_parquet(FACTS_PATH)
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    rank = {(logical, tag): i for logical, tags in CONCEPTS.items() for i, tag in enumerate(tags)}
    df["tag_rank"] = [rank.get((lg, tg), 99) for lg, tg in zip(df["logical"], df["tag"])]
    return df


def annual_rows(df):
    """Only annual-report figures: 10-K family, and for flow items a ~12-month duration."""
    df = df[df["form"].isin(ANNUAL_FORMS)]
    instant = df["logical"].isin(INSTANT_ITEMS)
    days = (df["end"] - df["start"]).dt.days
    return df[instant | days.between(*ANNUAL_DAYS)]


def known_annuals(df, as_of):
    """One row per (company, item, fiscal-period-end): the version known on `as_of`
    (latest filing on or before that date; preferred tag first)."""
    ann = annual_rows(df)
    ann = ann[ann["filed"] <= pd.Timestamp(as_of)]
    ann = ann.sort_values(["cik", "logical", "end", "tag_rank", "filed"], ascending=[True, True, True, True, False])
    return ann.drop_duplicates(["cik", "logical", "end"], keep="first")


def panel_as_of(df, as_of):
    """Wide table, one row per company: latest fiscal-year value of each item known on `as_of`,
    plus the prior year (`<item>_prior`) and the date it became public (`<item>_filed`)."""
    known = known_annuals(df, as_of)
    known = known.sort_values(["cik", "logical", "end"], ascending=[True, True, False])
    rank_in_group = known.groupby(["cik", "logical"]).cumcount()
    latest = known[rank_in_group == 0].set_index(["cik", "logical"])
    prior = known[rank_in_group == 1].set_index(["cik", "logical"])

    out = latest["val"].unstack("logical")
    out_prior = prior["val"].unstack("logical").add_suffix("_prior")
    out_end = latest["end"].unstack("logical").add_suffix("_end")
    out_filed = latest["filed"].unstack("logical").add_suffix("_filed")
    panel = pd.concat([out, out_prior, out_end, out_filed], axis=1)

    fy_end = panel[[c for c in panel.columns if c.endswith("_end")]].max(axis=1)
    panel["fy_end"] = fy_end
    stale = (pd.Timestamp(as_of) - fy_end).dt.days > MAX_STALENESS_DAYS
    return panel[~stale]


def restatement_stats(df, tolerance=0.01):
    """For each (company, item, fiscal year): was the value in the FIRST annual filing
    different (by more than `tolerance`) from the LAST filing's value? Measures how often
    a backtest using today's restated statements sees a different number than investors did."""
    ann = annual_rows(df)
    ann = ann[ann["logical"].isin(["revenue", "net_income", "total_assets", "equity", "cfo"])]
    ann = ann.sort_values(["cik", "logical", "end", "tag_rank", "filed"])
    best_rank = ann.groupby(["cik", "logical", "end"])["tag_rank"].transform("min")
    ann = ann[ann["tag_rank"] == best_rank]
    first = ann.groupby(["cik", "logical", "end"]).first()
    last = ann.groupby(["cik", "logical", "end"]).last()
    both = pd.DataFrame({"first": first["val"], "last": last["val"]}).dropna()
    both = both[both["first"] != 0]
    rel = (both["last"] - both["first"]).abs() / both["first"].abs()
    both["changed"] = rel > tolerance
    return both.groupby(level="logical")["changed"].agg(["mean", "sum", "count"])


def validate_against_yfinance(df, tickers):
    import yfinance as yf

    mapping = pd.read_csv(ROOT / "data" / "ticker_cik.csv", dtype={"cik": str})
    cik_of = dict(zip(mapping["ticker"], mapping["cik"]))
    panel = panel_as_of(df, pd.Timestamp.today())
    rows = []
    for t in tickers:
        cik = cik_of.get(t)
        if cik not in panel.index:
            rows.append((t, "no EDGAR panel row", None, None, None))
            continue
        try:
            fin = yf.Ticker(t).financials
            bs = yf.Ticker(t).balance_sheet
        except Exception as e:  # network/parse failure for one ticker shouldn't stop the check
            rows.append((t, f"yfinance error: {e}", None, None, None))
            continue
        for label, edgar_col, yf_frame, yf_key in (
            ("revenue", "revenue", fin, "Total Revenue"),
            ("net_income", "net_income", fin, "Net Income"),
            ("total_assets", "total_assets", bs, "Total Assets"),
        ):
            if yf_key not in yf_frame.index or edgar_col not in panel.columns:
                continue
            yf_val = yf_frame.loc[yf_key].dropna()
            if yf_val.empty:
                continue
            edgar_val = panel.loc[cik, edgar_col]
            rows.append((t, label, edgar_val, float(yf_val.iloc[0]),
                         (edgar_val - yf_val.iloc[0]) / abs(yf_val.iloc[0]) if yf_val.iloc[0] else np.nan))
    return pd.DataFrame(rows, columns=["ticker", "item", "edgar", "yfinance", "rel_diff"])


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "validate":
        print(__doc__)
        return
    df = load_facts()
    print("=== Restatement frequency: first-reported vs latest value, annual figures (>1% different) ===")
    stats = restatement_stats(df)
    print(stats.assign(share=lambda s: (s["mean"] * 100).round(1)).to_string())

    print("\n=== Point-in-time panel size at several dates ===")
    for d in ("2012-06-30", "2016-06-30", "2020-06-30", "2022-06-30", "2025-06-30"):
        p = panel_as_of(df, d)
        print(f"{d}: {len(p)} companies with a current fiscal year; revenue known for {p['revenue'].notna().sum()}")

    print("\n=== EDGAR (latest FY, today) vs yfinance (latest FY) ===")
    sample = ["AAPL", "MSFT", "JPM", "XOM", "WMT", "PG", "CAT", "NKE", "KO", "HD"]
    cmp = validate_against_yfinance(df, sample)
    print(cmp.to_string(index=False))
    ok = cmp["rel_diff"].dropna()
    print(f"\nmatches within 1%: {(ok.abs() < 0.01).sum()} of {len(ok)}; median |diff| = {ok.abs().median():.4f}")


if __name__ == "__main__":
    main()
