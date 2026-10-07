"""g3_diagnose.py -- RESEARCH SANDBOX. WHY did DCF-stability fail to predict returns? Diagnostic only (no candidate, no trial)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import factor_ic as fi
DATA = Path(__file__).resolve().parent / "data"
t_ = lambda x: (lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan)(pd.Series(x).dropna())
sc = pd.read_csv(DATA / "g3_scores.csv", parse_dates=["date"]); sc = sc[sc.error.isna() & (sc.date <= fi.DEV_END)].dropna(subset=["composite_score"])
adj = pd.read_parquet(DATA / "prices_adjclose.parquet"); idx = adj.index.to_series(); qe = sorted(idx.groupby(idx.index.to_period("Q")).max()); nxt = {d: qe[i+1] for i, d in enumerate(qe[:-1])}
sc = sc[sc.date.isin(nxt)].copy()
sc["fwd"] = [adj.at[nxt[d], t] / adj.at[d, t] - 1 if t in adj.columns else np.nan for d, t in zip(sc.date, sc.ticker)]
vol = adj.pct_change(fill_method=None).rolling(252, min_periods=200).std()
sc["vol"] = [vol.at[d, t] if t in vol.columns and d in vol.index else np.nan for d, t in zip(sc.date, sc.ticker)]
sc = sc.dropna(subset=["fwd"])
sc["absup"] = sc.upside_pct.abs()
sc["absdev"] = sc.fwd - sc.groupby("date").fwd.transform("mean"); sc["absdev"] = sc.absdev.abs()
def per_date(f, minn=100):
    return pd.Series({d: f(g) for d, g in sc.groupby("date") if len(g) >= minn})
sp = lambda a, b: (lambda g: g[a].corr(g[b], method="spearman"))

print("1) Is 'stability' just a proxy for how extreme the upside is?")
print(f"   mean per-date Spearman(S1, |upside|) = {per_date(sp('s1','absup')).mean():+.2f};  (S2, |upside|) = {per_date(sp('s2','absup')).mean():+.2f}")

print("\n2) Does ANY conviction level carry signal? composite IC by |upside| bin (per-date terciles-of-conviction -> pooled across dates)")
sc["conv"] = sc.groupby("date").absup.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
for b in range(5):
    ic = pd.Series({d: g.composite_score.corr(g.fwd, method="spearman") for d, g in sc[sc.conv == b].groupby("date") if len(g) >= 40})
    print(f"   conviction quintile {b} (0=lowest |upside|): IC {ic.mean():+.4f} (t {t_(ic):+.2f}, {len(ic)} dates)")

print("\n3) Sign asymmetry: realised next-quarter return (% , demeaned by date) by DCF verdict and stability")
sc["fwd_dm"] = (sc.fwd - sc.groupby("date").fwd.transform("mean")) * 100
sc["side"] = np.where(sc.upside_pct > 0, "DCF says cheap", "DCF says rich"); sc["stab"] = np.where(sc.s1 >= 0.8, "stable", "unstable")
tab = sc.groupby(["side", "stab"]).agg(n=("fwd_dm", "size"), mean_excess_pct=("fwd_dm", "mean")).round(2); print(tab.to_string())

print("\n4) Direction vs MAGNITUDE: does the Monte Carlo spread predict how MUCH a stock moves (risk), beyond trailing volatility?")
sc["w"] = sc.mc_rel_width
r_w = per_date(sp("w", "absdev")); r_v = per_date(sp("vol", "absdev")); r_wv = per_date(sp("w", "vol"))
print(f"   Spearman(MC width, |move|) = {r_w.mean():+.3f} (t {t_(r_w):+.1f}) | Spearman(trailing vol, |move|) = {r_v.mean():+.3f} (t {t_(r_v):+.1f}) | Spearman(MC width, trailing vol) = {r_wv.mean():+.2f}")
def partial(g):
    r = g[["w", "vol", "absdev"]].rank().dropna()
    if len(r) < 100: return np.nan
    res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
    return np.corrcoef(res(r.w, r.vol), res(r.absdev, r.vol))[0, 1]
pc = per_date(partial); print(f"   partial correlation of MC width with |move| AFTER removing trailing volatility: {pc.mean():+.3f} (t {t_(pc):+.1f}, {pc.notna().sum()} dates)")
hi = sc[sc.w >= sc.groupby('date').w.transform(lambda s: s.quantile(.8))]; lo = sc[sc.w <= sc.groupby('date').w.transform(lambda s: s.quantile(.2))]
print(f"   realised |move| (% of stock vs market): widest-MC-quintile {hi.absdev.mean()*100:.1f}% vs narrowest {lo.absdev.mean()*100:.1f}%")
