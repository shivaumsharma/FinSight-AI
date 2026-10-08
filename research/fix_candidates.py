"""fix_candidates.py -- RESEARCH SANDBOX. Scores the FIX-CANDIDATE PRE-REGISTRATION in SPRINT_TRACKER.md exactly."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import factor_ic as fi

DATA = Path(__file__).resolve().parent / "data"
SEG_SPLIT = pd.Timestamp("2018-12-31")
MIN_NAMES, MIN_ON = 100, 15

def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan

sc = pd.read_csv(DATA / "p2b_scores.csv", parse_dates=["date"])
sc = sc[sc["error"].isna() & (sc["date"] <= fi.DEV_END)].dropna(subset=["composite_score"])
adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
idx = adj.index.to_series(); qe = sorted(idx.groupby(idx.index.to_period("Q")).max()); nxt = {d: qe[i+1] for i, d in enumerate(qe[:-1])}
sc = sc[sc["date"].isin(nxt)].copy()
sc["fwd"] = [adj.at[nxt[d], t] / adj.at[d, t] - 1 if t in adj.columns else np.nan for d, t in zip(sc["date"], sc["ticker"])]
def fwd4(d, t):
    j = qe.index(d) + 4
    return adj.at[qe[j], t] / adj.at[d, t] - 1 if j < len(qe) and t in adj.columns else np.nan
sc["fwd4"] = [fwd4(d, t) for d, t in zip(sc["date"], sc["ticker"])]

data = fi.load_all(); parts, spread = [], {}
for d in sorted(sc["date"].unique()):
    f = fi.add_combos(fi.factors_at(data, d))
    g = sc[sc["date"] == d].set_index("ticker").join(f[["mom_12_1", "roa", "earnings_yield", "valuation_only"]], how="left")
    r = lambda c: g[c].rank(pct=True)
    g["C0_composite"] = g["composite_score"]
    g["C1_dcf_rank"] = r("upside_pct")
    g["C2a_comp+mom"] = (r("composite_score") + r("mom_12_1")) / 2
    g["C2b_comp+roa"] = (r("composite_score") + r("roa")) / 2
    g["C2c_comp+mom+roa"] = (r("composite_score") + r("mom_12_1") + r("roa")) / 3
    v = g["valuation_only"].dropna().index
    ey = g.loc[v, "earnings_yield"]; q = pd.qcut(g.loc[v, "valuation_only"].rank(method="first"), 5, labels=False)
    spread[d] = ey[q == 4].median() - ey[q == 0].median()
    g["date"] = d; parts.append(g.reset_index())
P = pd.concat(parts)
sp = pd.Series(spread).sort_index()
on = {d: (i >= 12 and sp.iloc[i] > sp.iloc[:i].median()) for i, d in enumerate(sp.index)}

def ics(col, hor="fwd", dates=None):
    out = {}
    for d, g in P.groupby("date"):
        if dates is not None and d not in dates: continue
        g = g.dropna(subset=[col, hor])
        if len(g) >= MIN_NAMES: out[d] = g[col].corr(g[hor], method="spearman")
    return pd.Series(out)

rows = []
for c in ["C0_composite", "C1_dcf_rank", "C2a_comp+mom", "C2b_comp+roa", "C2c_comp+mom+roa"]:
    ic = ics(c); a, b = ic[ic.index <= SEG_SPLIT], ic[ic.index > SEG_SPLIT]; i4 = ics(c, "fwd4")
    ok = ic.mean() >= 0.03 and tstat(ic) >= 2.5 and a.mean() > 0 and b.mean() > 0
    rows.append({"candidate": c, "pooled_IC": ic.mean(), "t": tstat(ic), "segA_IC": a.mean(), "segB_IC": b.mean(), "IC_4q(desc)": i4.mean(),
                 "PASS": "baseline" if c == "C0_composite" else ok})
on_dates = {d for d, v in on.items() if v}
ic3 = ics("C0_composite", dates=on_dates); off3 = ics("C0_composite", dates={d for d, v in on.items() if not v and d in sp.index[12:]})
a, b = ic3[ic3.index <= SEG_SPLIT], ic3[ic3.index > SEG_SPLIT]
rows.append({"candidate": "C3_regime_ON_dates", "pooled_IC": ic3.mean(), "t": tstat(ic3), "segA_IC": a.mean(), "segB_IC": b.mean(),
             "IC_4q(desc)": ics("C0_composite", "fwd4", on_dates).mean(),
             "PASS": bool(len(ic3) >= MIN_ON and ic3.mean() >= 0.03 and tstat(ic3) >= 2.5 and a.mean() > 0 and b.mean() > 0)})
res = pd.DataFrame(rows).set_index("candidate")
pd.set_option("display.width", 200)
print(res.round(4).to_string())
print(f"\nCandidate 3: {len(on_dates)} ON dates, {len(off3)} OFF dates; OFF-date IC {off3.mean():+.4f} (t {tstat(off3):+.2f}); "
      f"value-spread vs IC correlation across all dates: {np.corrcoef(sp.reindex(ics('C0_composite').index), ics('C0_composite'))[0,1]:+.2f}")
print("Mean spread by year:", {y: round(v, 3) for y, v in sp.groupby(sp.index.year).mean().items()})
res.to_csv(DATA / "fix_candidates_result.csv")
