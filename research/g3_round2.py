"""g3_round2.py -- RESEARCH SANDBOX. Scores the G3 ROUND-2 PRE-REGISTRATION in SPRINT_TRACKER.md exactly."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import factor_ic as fi
DATA = Path(__file__).resolve().parent / "data"
BUY, SPLIT, MIN_G = 7.5, pd.Timestamp("2018-12-31"), 20
t_ = lambda x: (lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 and s.std(ddof=1) > 0 else np.nan)(pd.Series(x).dropna())
sc = pd.read_csv(DATA / "g3_scores.csv", parse_dates=["date"]); sc = sc[sc.error.isna() & (sc.date <= fi.DEV_END)].dropna(subset=["composite_score"])
adj = pd.read_parquet(DATA / "prices_adjclose.parquet"); idx = adj.index.to_series(); qe = sorted(idx.groupby(idx.index.to_period("Q")).max()); nxt = {d: qe[i+1] for i, d in enumerate(qe[:-1])}
sc = sc[sc.date.isin(nxt)].copy()
sc["fwd"] = [adj.at[nxt[d], t] / adj.at[d, t] - 1 if t in adj.columns else np.nan for d, t in zip(sc.date, sc.ticker)]
sc = sc.dropna(subset=["fwd"]); sc["ex"] = (sc.fwd - sc.groupby("date").fwd.transform("mean")) * 100
halves = lambda s: (s[s.index <= SPLIT].mean(), s[s.index > SPLIT].mean())

# G3b
rows = {}
for d, g in sc.groupby("date"):
    b = g[g.composite_score >= BUY]; st, un = b[b.s1 >= 0.8], b[b.s1 < 0.8]
    if len(st) >= MIN_G and len(un) >= MIN_G: rows[d] = (st.ex.mean() - un.ex.mean(), st.ex.mean(), un.ex.mean(), b.ex.mean())
r = pd.DataFrame(rows, index=["diff", "stable", "unstable", "all_buy"]).T
a, b2 = halves(r["diff"])
ok_b = len(r) >= 30 and r["diff"].mean() >= 0.5 and t_(r["diff"]) >= 2.5 and a > 0 and b2 > 0
print(f"G3b Buy&stable minus Buy&unstable: {r['diff'].mean():+.2f} pts/qtr (t {t_(r['diff']):+.2f}, {len(r)} dates) | halves {a:+.2f} / {b2:+.2f} | PASS: {bool(ok_b)}")
print(f"    mean excess: Buy&stable {r.stable.mean():+.2f}, Buy&unstable {r.unstable.mean():+.2f}, all Buy {r.all_buy.mean():+.2f} %/qtr")

# G3c
sc["absup"] = sc.upside_pct.abs(); sc["q"] = sc.groupby("date").absup.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
ic = pd.Series({d: g.composite_score.corr(g.fwd, method="spearman") for d, g in sc[sc.q < 4].groupby("date") if len(g) >= 100})
a, b2 = halves(ic); ok_c = ic.mean() >= 0.03 and t_(ic) >= 2.5 and a > 0 and b2 > 0
print(f"G3c composite IC excluding the top-|upside| quintile: {ic.mean():+.4f} (t {t_(ic):+.2f}, {len(ic)} dates) | halves {a:+.4f} / {b2:+.4f} | PASS: {bool(ok_c)}")
print("\nPRE-REGISTERED VERDICT:", {"G3b": bool(ok_b), "G3c": bool(ok_c)})
