"""
sue_ic.py -- RESEARCH SANDBOX. Scores the SUE PRE-REGISTRATION in SPRINT_TRACKER.md exactly.

SUE = (EPS_q - EPS_{q-4}) / std of the last 8 year-over-year EPS changes (>= 6 needed), stamped with the filing date.
At each quarter-end T, a company's SUE is its latest one filed before T and within the previous 100 days.
S1 (this file): per-quarter Spearman IC of SUE vs next-quarter total return. S2 lives in ml_cross_section_4q_sue.py.

Run: python research/sue_ic.py        (after research/eps_pull.py)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import factor_ic as fi  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
MAX_AGE_DAYS, MIN_NAMES, SPLIT = 100, 100, pd.Timestamp("2018-12-31")


def quarterly_eps(g):
    """One company's quarterly series: Q1-Q3 from 10-Qs, Q4 = annual - (Q1+Q2+Q3). Columns: end, filed, eps."""
    q = g[g.kind == "Q"][["end", "filed", "eps"]]
    rows = [q]
    for a in g[g.kind == "A"].itertuples():
        inside = q[(q.end > a.end - pd.Timedelta(days=300)) & (q.end < a.end - pd.Timedelta(days=30))]
        if len(inside) == 3:
            rows.append(pd.DataFrame({"end": [a.end], "filed": [a.filed], "eps": [a.eps - inside.eps.sum()]}))
    return pd.concat(rows).sort_values("end").drop_duplicates("end").reset_index(drop=True)


def sue_series(g):
    s = quarterly_eps(g)
    if len(s) < 9:
        return pd.DataFrame(columns=["filed", "sue"])
    d = []
    for r in s.itertuples():
        match = s[(s.end - (r.end - pd.Timedelta(days=365))).abs() <= pd.Timedelta(days=20)]
        d.append(r.eps - match.eps.iloc[0] if len(match) else np.nan)
    s["d"] = d
    sd = s["d"].rolling(8, min_periods=6).std()
    s["sue"] = s["d"] / sd.replace(0, np.nan)
    return s.dropna(subset=["sue"])[["filed", "sue"]]


def sue_panel(dates, tickers_by_cik):
    """DataFrame (date, ticker, sue) of each company's latest SUE filed before `date` and within MAX_AGE_DAYS."""
    eps = pd.read_parquet(DATA / "eps_quarterly.parquet")
    series = {cik: sue_series(g) for cik, g in eps.groupby("cik")}
    out = []
    for d in dates:
        for cik, s in series.items():
            known = s[(s.filed < d) & (s.filed >= d - pd.Timedelta(days=MAX_AGE_DAYS))]
            if len(known):
                for t in tickers_by_cik.get(cik, []):
                    out.append((d, t, float(known.sort_values("filed").sue.iloc[-1])))
    return pd.DataFrame(out, columns=["date", "ticker", "sue"])


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def main():
    m = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str}).drop_duplicates("ticker")
    by_cik = m.groupby("cik")["ticker"].apply(list).to_dict()
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    idx = adj.index.to_series()
    qe = [d for d in sorted(idx.groupby(idx.index.to_period("Q")).max()) if d >= fi.FIRST_DATE]
    nxt = {d: qe[i + 1] for i, d in enumerate(qe[:-1])}
    dates = [d for d in qe if d <= fi.DEV_END and d in nxt]
    panel = sue_panel(dates, by_cik)
    panel["fwd"] = [adj.at[nxt[d], t] / adj.at[d, t] - 1 if t in adj.columns else np.nan for d, t in zip(panel.date, panel.ticker)]
    panel = panel.dropna(subset=["fwd"])
    print(f"SUE available for {panel.ticker.nunique()} tickers; names per date: median {int(panel.groupby('date').size().median())}")
    ic = pd.Series({d: g.sue.corr(g.fwd, method="spearman") for d, g in panel.groupby("date") if len(g) >= MIN_NAMES})
    a, b = ic[ic.index <= SPLIT].mean(), ic[ic.index > SPLIT].mean()
    ok = ic.mean() >= 0.02 and tstat(ic) >= 2.5 and a > 0 and b > 0
    print(f"S1 SUE vs next-quarter return: mean IC {ic.mean():+.4f} | t {tstat(ic):+.2f} | positive {(ic > 0).mean():.0%} | n={len(ic)} | halves {a:+.4f} / {b:+.4f} | PASS: {bool(ok)}")
    print("   by year:", {y: round(v, 3) for y, v in ic.groupby(ic.index.year).mean().items()})
    qs = []
    for d, g in panel.groupby("date"):
        if len(g) >= MIN_NAMES:
            q = pd.qcut(g.sue.rank(method="first"), 5, labels=False)
            qs.append(g.fwd[q == 4].mean() - g.fwd[q == 0].mean())
    print(f"   (descriptive) top-minus-bottom SUE quintile: {np.mean(qs) * 100:+.2f}%/qtr, t {tstat(qs):+.2f}")


if __name__ == "__main__":
    main()
