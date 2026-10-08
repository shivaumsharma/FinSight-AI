"""ml_cross_section_4q.py -- RESEARCH SANDBOX. A1b of the G5 pre-registration in SPRINT_TRACKER.md, exactly as written:
the A1 walk-forward models, but target and evaluation are the next FOUR quarters' return; Newey-West (lag 3) t-statistics."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import ml_cross_section as m

DATA, H, LAG = m.DATA, 4, 3

def nw_t(x, lag=LAG):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x); mu = x.mean(); e = x - mu
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return mu / np.sqrt(v / n)

def main():
    P, _ = m.build_panel(); P = P.reset_index(drop=True)
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet"); idx = adj.index.to_series()
    qe = [d for d in sorted(idx.groupby(idx.index.to_period("Q")).max()) if d >= m.fi.FIRST_DATE]
    end_of = {d: qe[i + H] for i, d in enumerate(qe) if i + H < len(qe)}
    P["fwd4"] = [adj.at[end_of[d], t] / adj.at[d, t] - 1 if d in end_of and t in adj.columns else np.nan for d, t in zip(P.date, P.ticker)]
    P["y"] = P.groupby("date")["fwd4"].rank(pct=True); X = m.prep(P)
    tests = [d for d in sorted(P.date.unique()) if d >= m.FIRST_TEST and d in end_of]
    print(f"rows {len(P)}, test dates {len(tests)}, features {X.shape[1]}")
    preds = {k: pd.Series(np.nan, index=P.index) for k in ("hgb", "ridge")}; model = {}
    for i, T in enumerate(tests):
        if i % m.RETRAIN_EVERY == 0:
            tr = P.date.map(lambda d: d in end_of and end_of[d] <= T) & P.y.notna()
            model["hgb"] = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(X[tr], P.loc[tr, "y"])
            model["ridge"] = Ridge(alpha=10).fit(X[tr], P.loc[tr, "y"])
            print(f"   retrain at {T.date()}: {int(tr.sum())} rows", flush=True)
        te = P.date == T
        for k in model: preds[k][te] = model[k].predict(X[te])
    dump = P.loc[P.date.isin(tests), ["date", "ticker", "fwd", "fwd4"]].copy()
    dump["pred_hgb"], dump["pred_ridge"] = preds["hgb"][dump.index], preds["ridge"][dump.index]
    dump.to_parquet(DATA / "a1b_predictions.parquet")        # walk-forward predictions, for the portfolio check
    for k in preds:
        ic = pd.Series({T: preds[k][P.date == T].corr(P.loc[P.date == T, "fwd4"], method="spearman") for T in tests if ((P.date == T) & P.fwd4.notna()).sum() >= m.MIN_NAMES})
        ok = ic.mean() >= 0.03 and nw_t(ic) >= 2.5
        print(f"A1b {k:6s} 4q-horizon mean IC {ic.mean():+.4f} | naive t {ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic))):+.2f} | Newey-West t {nw_t(ic):+.2f} | n={len(ic)} | PASS: {bool(ok)}")
        print("      by year:", {y: round(v, 3) for y, v in ic.groupby(ic.index.year).mean().items()})
    comp = pd.Series({T: (0.8 * g.dcf_score + 0.2 * g.relative_score).corr(g.fwd4, method="spearman") for T, g in P[P.date.isin(tests)].dropna(subset=["dcf_score", "relative_score", "fwd4"]).groupby("date") if len(g) >= m.MIN_NAMES})
    print(f"production composite, same dates, 4q horizon: mean IC {comp.mean():+.4f} (Newey-West t {nw_t(comp):+.2f})")

if __name__ == "__main__":
    main()
