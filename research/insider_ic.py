"""
insider_ic.py -- RESEARCH SANDBOX. Scores the INSIDER PRE-REGISTRATION in SPRINT_TRACKER.md exactly.

I1 = open-market purchase value by officers/directors in the 180 days before T, divided by market cap at T.
I2 = number of distinct insiders who bought in the same window. Stocks with no purchase get 0.
`insider_panel(dates, tickers_by_cik)` is also used by ml_cross_section_4q_insider.py for the feature test.

Run: python research/insider_ic.py       (after research/insider_pull.py)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import delisting  # noqa: E402
from research import factor_ic as fi  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
WINDOW_DAYS, MIN_NAMES, SPLIT = 180, 100, pd.Timestamp("2018-12-31")


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def insider_panel(dates, tickers_by_cik, mcap_by_date=None):
    """(date, ticker, i1_value, i2_buyers). i1_value is the raw dollar value (divide by market cap where it is known)."""
    ib = pd.read_parquet(DATA / "insider_buys.parquet")
    out = []
    for d in dates:
        w = ib[(ib.filed < d) & (ib.filed >= d - pd.Timedelta(days=WINDOW_DAYS))]
        g = w.groupby("cik").agg(value=("value", "sum"), buyers=("owner", "nunique"))
        for cik, r in g.iterrows():
            for t in tickers_by_cik.get(cik, []):
                out.append((d, t, float(r.value), int(r.buyers)))
    return pd.DataFrame(out, columns=["date", "ticker", "i_value", "i_buyers"])


def main():
    m = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str}).drop_duplicates("ticker")
    by_cik = m.groupby("cik")["ticker"].apply(list).to_dict()
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    idx = adj.index.to_series()
    qe = [d for d in sorted(idx.groupby(idx.index.to_period("Q")).max()) if d >= fi.FIRST_DATE]
    nxt = {d: qe[i + 1] for i, d in enumerate(qe[:-1])}
    dates = [d for d in qe if d <= fi.DEV_END and d in nxt]
    data = fi.load_all()
    purchases = insider_panel(dates, by_cik)
    rows = []
    for d in dates:
        f = fi.factors_at(data, d)                                    # market cap at d (delisted names masked as in every other test)
        g = pd.DataFrame({"ticker": f.index, "mcap": f["mcap"].values, "date": d})
        g = g[g.mcap.notna() & (g.mcap > 0)]
        g = g.merge(purchases[purchases.date == d], on=["date", "ticker"], how="left").fillna({"i_value": 0.0, "i_buyers": 0})
        g["i1"] = g.i_value / g.mcap
        g["fwd"] = [adj.at[nxt[d], t] / adj.at[d, t] - 1 if t in adj.columns else np.nan for t in g.ticker]
        rows.append(g.dropna(subset=["fwd"]))
    P = pd.concat(rows)
    share = (P.i_buyers > 0).mean()
    print(f"panel rows {len(P)}, dates {P.date.nunique()}, share of stock-dates with any insider purchase in the prior 180 days: {share:.1%}")
    for name, col in (("I1 purchase value / market cap", "i1"), ("I2 distinct insider buyers", "i_buyers")):
        ic = pd.Series({d: g[col].corr(g.fwd, method="spearman") for d, g in P.groupby("date") if len(g) >= MIN_NAMES})
        a, b = ic[ic.index <= SPLIT].mean(), ic[ic.index > SPLIT].mean()
        ok = ic.mean() >= 0.02 and tstat(ic) >= 2.5 and a > 0 and b > 0
        print(f"{name}: mean IC {ic.mean():+.4f} | t {tstat(ic):+.2f} | positive {(ic > 0).mean():.0%} | n={len(ic)} | halves {a:+.4f} / {b:+.4f} | PASS: {bool(ok)}")
        print("   by year:", {y: round(v, 3) for y, v in ic.groupby(ic.index.year).mean().items()})
    has = P[P.i_buyers > 0].groupby("date").fwd.mean(); none = P[P.i_buyers == 0].groupby("date").fwd.mean()
    diff = (has - none).dropna()
    print(f"(descriptive) next-quarter return, stocks WITH a purchase minus WITHOUT: {diff.mean() * 100:+.2f}%/qtr (t {tstat(diff):+.2f}, {len(diff)} dates)")


if __name__ == "__main__":
    main()
