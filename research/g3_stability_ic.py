"""g3_stability_ic.py -- RESEARCH SANDBOX. Scores the G3 PRE-REGISTRATION in SPRINT_TRACKER.md exactly.

Stable = S >= 0.8, unstable = S < 0.8 (S1 = Monte Carlo agreement with the base verdict, S2 = share of the 25 WACC x
terminal-growth cells agreeing). S3 = Monte Carlo prob_undervalued used directly as a score.
Pass: stable-group IC >= 0.03 & t >= 2.5 AND paired (stable - unstable) t >= 2 [S1, S2]; S3: IC >= 0.03 & t >= 2.5.
Dates need >= 100 scored names; the stable group needs >= 40 names and >= 30 dates."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import factor_ic as fi  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
THRESH, MIN_NAMES, MIN_STABLE, MIN_DATES = 0.8, 100, 40, 30


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def main():
    sc = pd.read_csv(DATA / "g3_scores.csv", parse_dates=["date"])
    sc = sc[sc["error"].isna() & (sc["date"] <= fi.DEV_END)].dropna(subset=["composite_score"])
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    idx = adj.index.to_series()
    qe = sorted(idx.groupby(idx.index.to_period("Q")).max())
    nxt = {d: qe[i + 1] for i, d in enumerate(qe[:-1])}
    sc = sc[sc["date"].isin(nxt)].copy()
    sc["fwd"] = [adj.at[nxt[d], t] / adj.at[d, t] - 1 if t in adj.columns else np.nan for d, t in zip(sc["date"], sc["ticker"])]
    sc = sc.dropna(subset=["fwd"])
    print(f"scored rows {len(sc)}; with S1 {sc['s1'].notna().sum()}, S2 {sc['s2'].notna().sum()}, S3 {sc['s3'].notna().sum()}")
    print(f"share stable (>=0.8): S1 {(sc['s1'] >= THRESH).mean():.0%}, S2 {(sc['s2'] >= THRESH).mean():.0%}")

    def ic(g, col="composite_score"):
        return g[col].corr(g["fwd"], method="spearman") if len(g) >= MIN_STABLE and g[col].nunique() > 2 else np.nan

    results = {}
    for s in ("s1", "s2"):
        stab, unst, allv = {}, {}, {}
        for d, g in sc.dropna(subset=[s]).groupby("date"):
            if len(g) < MIN_NAMES:
                continue
            allv[d] = ic(g)
            stab[d] = ic(g[g[s] >= THRESH])
            unst[d] = ic(g[g[s] < THRESH])
        st, un, al = pd.Series(stab), pd.Series(unst), pd.Series(allv)
        both = pd.concat([st, un], axis=1, keys=["st", "un"]).dropna()
        diff = both["st"] - both["un"]
        ok = (st.notna().sum() >= MIN_DATES and st.mean() >= 0.03 and tstat(st) >= 2.5 and tstat(diff) >= 2)
        results[s] = ok
        print(f"\n{s.upper()}: stable-group IC {st.mean():+.4f} (t {tstat(st):+.2f}, {st.notna().sum()} dates) | "
              f"unstable {un.mean():+.4f} (t {tstat(un):+.2f}) | all {al.mean():+.4f} | paired stable-unstable {diff.mean():+.4f} (t {tstat(diff):+.2f}, n={len(diff)}) | PASS: {bool(ok)}")
        print("   stable IC by year:", {y: round(v, 3) for y, v in st.groupby(st.index.year).mean().items()})
    s3 = pd.Series({d: ic(g, "s3") for d, g in sc.dropna(subset=["s3"]).groupby("date") if len(g) >= MIN_NAMES})
    ok3 = s3.mean() >= 0.03 and tstat(s3) >= 2.5
    results["s3"] = ok3
    print(f"\nS3 (MC prob_undervalued as score): IC {s3.mean():+.4f} (t {tstat(s3):+.2f}, {s3.notna().sum()} dates) | PASS: {bool(ok3)}")
    print("\nPRE-REGISTERED VERDICT:", {k: bool(v) for k, v in results.items()})


if __name__ == "__main__":
    main()
