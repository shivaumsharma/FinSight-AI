"""
pit_composite_ic.py -- RESEARCH SANDBOX. Scores the P2b pre-registration in SPRINT_TRACKER.md.

H1: Spearman IC of raw composite_score vs next-quarter total return, per quarter-end.
H2: sector-neutral -- within-(date, sector) percentile rank of composite_score vs forward
    return minus the (date, sector) mean return.
Pass line: mean IC >= 0.02 and t >= 2 across dates. Decisive window: 2012-06 .. 2022-12
(out-of-sample for the hint found on 2024-06 .. 2026-06). Dates need >= 100 scored names.

Run: python research/pit_composite_ic.py   (after research/pit_composite.py run)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
DEV_END = pd.Timestamp("2025-06-30")
OOS_END = pd.Timestamp("2022-12-31")
MIN_NAMES = 100
MIN_SECTOR = 5


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def main():
    sc = pd.read_csv(DATA / "p2b_scores.csv", parse_dates=["date"])
    errors = sc["error"].notna().sum() if "error" in sc else 0
    sc = sc[sc["error"].isna()] if "error" in sc else sc
    sc = sc[sc["date"] <= DEV_END].dropna(subset=["composite_score"])
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    idx = adj.index.to_series()
    qe = sorted(idx.groupby(idx.index.to_period("Q")).max())
    nxt = {d: qe[i + 1] for i, d in enumerate(qe[:-1])}
    cat = pd.read_csv(DATA / "ticker_cik.csv").set_index("ticker")["category"]
    sc["sector"] = sc["ticker"].map(cat).str.replace(r" \(S&P \d+\)", "", regex=True)
    sc = sc[sc["date"].isin(nxt)]
    sc["fwd"] = [adj.at[nxt[d], t] / adj.at[d, t] - 1 if t in adj.columns else np.nan for d, t in zip(sc["date"], sc["ticker"])]
    sc = sc.dropna(subset=["fwd"])
    sc["rank_in_sector"] = sc.groupby(["date", "sector"])["composite_score"].rank(pct=True)
    sc["fwd_sec"] = sc["fwd"] - sc.groupby(["date", "sector"])["fwd"].transform("mean")
    sc["n_sector"] = sc.groupby(["date", "sector"])["fwd"].transform("size")

    h1, h2, n = {}, {}, {}
    for d, g in sc.groupby("date"):
        if len(g) < MIN_NAMES:
            continue
        n[d] = len(g)
        h1[d] = g["composite_score"].corr(g["fwd"], method="spearman")
        gs = g[g["n_sector"] >= MIN_SECTOR]
        h2[d] = gs["rank_in_sector"].corr(gs["fwd_sec"], method="spearman")
    ic = pd.DataFrame({"n": pd.Series(n), "H1_raw": pd.Series(h1), "H2_sector_neutral": pd.Series(h2)})
    ic["diff"] = ic["H2_sector_neutral"] - ic["H1_raw"]
    ic.to_csv(DATA / "p2b_ic_by_date.csv")

    print(f"scored ticker-dates: {len(sc)} ({errors} errors dropped); dates kept (>= {MIN_NAMES} names): {len(ic)}; "
          f"names per date: median {int(ic['n'].median())}, min {int(ic['n'].min())}, max {int(ic['n'].max())}")
    for label, sub in (("OUT-OF-SAMPLE 2012-06 .. 2022-12 (decisive)", ic[ic.index <= OOS_END]),
                       ("LATER 2023-03 .. 2025-06", ic[ic.index > OOS_END]),
                       ("ALL DEVELOPMENT DATES", ic)):
        print(f"\n=== {label}: {len(sub)} dates ===")
        for col in ("H1_raw", "H2_sector_neutral"):
            x = sub[col].dropna()
            print(f"  {col:18s} mean IC {x.mean():+.4f} | std {x.std(ddof=1):.3f} | t {tstat(x):+.2f} | positive {int((x > 0).sum())}/{len(x)}"
                  f" | alive (IC>=0.02 & t>=2): {x.mean() >= 0.02 and tstat(x) >= 2}")
        d = sub["diff"].dropna()
        print(f"  paired H2-H1       mean {d.mean():+.4f} | t {tstat(d):+.2f}")
    print("\nMean IC by year (H1 / H2):")
    print(ic[["H1_raw", "H2_sector_neutral"]].groupby(ic.index.year).mean().round(3).T.to_string())

    # Descriptive only: top-minus-bottom quintile, and Buy-vs-rest, on the decisive window
    oos = sc[sc["date"] <= OOS_END]
    sp = []
    for d, g in oos.groupby("date"):
        if len(g) < MIN_NAMES:
            continue
        q = pd.qcut(g["composite_score"].rank(method="first"), 5, labels=False)
        sp.append(g["fwd"][q == 4].mean() - g["fwd"][q == 0].mean())
    print(f"\n(descriptive) out-of-sample raw Q5-Q1 spread: {np.mean(sp) * 100:+.2f}%/qtr, t {tstat(sp):+.2f}, n={len(sp)}")


if __name__ == "__main__":
    main()
