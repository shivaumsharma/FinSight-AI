"""why_composite_fails.py -- RESEARCH SANDBOX. Diagnostic (not a candidate strategy, no trials added):
why does the production composite have ~zero IC on EDGAR history? Uses p2b_scores.csv, development dates only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import factor_ic as fi

DATA = Path(__file__).resolve().parent / "data"
t_ = lambda x: (lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan)(pd.Series(x).dropna())

sc = pd.read_csv(DATA / "p2b_scores.csv", parse_dates=["date"])
sc = sc[sc["error"].isna() & (sc["date"] <= fi.DEV_END)].dropna(subset=["composite_score"])
adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
idx = adj.index.to_series(); qe = sorted(idx.groupby(idx.index.to_period("Q")).max()); nxt = {d: qe[i+1] for i, d in enumerate(qe[:-1])}
sc = sc[sc["date"].isin(nxt)]
def fwd(d, t, k=1):
    j = qe.index(d) + k
    return adj.at[qe[j], t] / adj.at[d, t] - 1 if j < len(qe) and t in adj.columns else np.nan
sc["fwd"] = [fwd(d, t) for d, t in zip(sc["date"], sc["ticker"])]
sc["fwd4"] = [fwd(d, t, 4) for d, t in zip(sc["date"], sc["ticker"])]
cat = pd.read_csv(DATA / "ticker_cik.csv").set_index("ticker")["category"]
sc["sector"] = sc["ticker"].map(cat).str.replace(r" \(S&P \d+\)", "", regex=True)

def ic_by_date(df, s, r="fwd", minn=100):
    out = {d: g[s].corr(g[r], method="spearman") for d, g in df.dropna(subset=[s, r]).groupby("date") if len(g) >= minn}
    return pd.Series(out)
def show(lab, ic): print(f"  {lab:34s} mean IC {ic.mean():+.4f}  t {t_(ic):+.2f}  n={len(ic)}")

print("1) COMPONENTS (all dev dates)")
for c in ("composite_score", "dcf_score", "relative_score"): show(c, ic_by_date(sc, c))

print("\n2) SCORE SHAPE")
for c in ("dcf_score", "relative_score"):
    x = sc[c].dropna(); print(f"  {c}: n={len(x)} | share >= +90: {(x >= 90).mean():.0%} | share <= -90: {(x <= -90).mean():.0%} | std {x.std():.0f}")
print(f"  DCF upside% percentiles 5/25/50/75/95: {np.nanpercentile(sc['upside_pct'], [5,25,50,75,95]).round(0)}")
print(f"  relative_score missing: {sc['relative_score'].isna().mean():.0%} of rows")

print("\n3) STABILITY: quarter-to-quarter rank autocorrelation of the same stock's score")
p = sc.pivot(index="date", columns="ticker", values="composite_score")
ac = [p.iloc[i].corr(p.iloc[i+1], method="spearman") for i in range(len(p)-1)]
print(f"  mean rank autocorr of composite: {np.nanmean(ac):.2f} (a slow-moving value score is usually 0.7+)")
pd_ = sc.pivot(index="date", columns="ticker", values="dcf_score")
print(f"  dcf_score: {np.nanmean([pd_.iloc[i].corr(pd_.iloc[i+1], method='spearman') for i in range(len(pd_)-1)]):.2f}")

print("\n4) DECILES of composite (pooled forward 1q return, then per-date mean)")
sc["dec"] = sc.groupby("date")["composite_score"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
dm = sc.groupby(["date", "dec"])["fwd"].mean().unstack().mean() * 100
print("  decile (0=lowest score): " + "  ".join(f"{i}:{v:+.1f}" for i, v in dm.items()) + "  (% per quarter)")

print("\n5) HORIZON: IC against the next 4 quarters (overlapping, descriptive only)")
show("composite vs 1q", ic_by_date(sc, "composite_score")); show("composite vs 4q", ic_by_date(sc, "composite_score", "fwd4"))
show("dcf_score vs 4q", ic_by_date(sc, "dcf_score", "fwd4"))

print("\n6) HIDDEN STYLE EXPOSURE: mean per-date Spearman of composite with known factors, and IC of those factors")
data = fi.load_all(); rows = []
for d in sorted(sc["date"].unique()):
    f = fi.add_combos(fi.factors_at(data, d)); g = sc[sc["date"] == d].set_index("ticker")
    f = f.reindex(g.index); f["mcap"] = np.log(f["mcap"])
    rows.append({"date": d, **{k: g["composite_score"].corr(f[k], method="spearman") for k in fi.FACTORS + ["mcap"]}})
ex = pd.DataFrame(rows).set_index("date"); print("  " + ex.mean().round(3).to_string().replace("\n", "\n  "))

print("\n7) BY SECTOR (IC of composite, dates with >=15 names)")
for s_, g in sc.groupby("sector"):
    ic = ic_by_date(g, "composite_score", minn=15)
    print(f"  {s_:26s} mean IC {ic.mean():+.3f}  t {t_(ic):+.2f}  ({len(g)} rows)")

print("\n8) NOISE vs SIGNAL: per-date IC std vs what sampling noise alone gives")
ic = ic_by_date(sc, "composite_score"); nn = sc.groupby("date").size().reindex(ic.index)
print(f"  observed std of per-date IC {ic.std():.3f}; pure sampling noise at n~{int(nn.median())} would be ~{(1/np.sqrt(nn)).mean():.3f}")
print("  years: " + "  ".join(f"{y}:{v:+.2f}" for y, v in ic.groupby(ic.index.year).mean().items()))
