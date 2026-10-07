"""
holdout_run.py -- RESEARCH SANDBOX. The ONE-SHOT sealed-holdout test of the frozen learned model, implemented exactly as the
"SEALED HOLDOUT -- PRE-REGISTRATION" in SPRINT_TRACKER.md (committed before this file was run).

  python research/holdout_run.py score   # compute the production model's scores for the 4 holdout dates (inputs only, no results)
  python research/holdout_run.py eval    # apply the FROZEN model and evaluate by the pre-registered rule (consumes the holdout)

Guards: verifies the frozen model file's SHA-256, never retrains, never refits, refuses to run `eval` a second time.
"""
import hashlib
import json
import sys
import time
import warnings
from math import sqrt
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import factor_ic as fi  # noqa: E402
from research import ml_cross_section as m  # noqa: E402
from research import pit_composite as pc  # noqa: E402

DATA = pc.DATA
HOLDOUT_DATES = [pd.Timestamp(x) for x in ("2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30")]
FROZEN_SHA = "ec8b1e25e19ca3d803b2fe157419ff9600a09b1b29ea9c60a1cdb1261507791d"
COST_BPS = 25
RESULT = DATA / "holdout_result.json"


def score():
    jobs = pc.prepare()
    market = pc.market_series()
    t0 = time.time()
    with Pool(16, initializer=pc._init_worker, initargs=(market,)) as pool:
        rows = [r for chunk in pool.imap_unordered(pc.score_ticker, [(j, HOLDOUT_DATES) for j in jobs], chunksize=1) for r in chunk]
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "p2b_scores_holdout.csv", index=False)
    base = pd.read_csv(DATA / "p2b_scores.csv")
    pd.concat([base, out]).to_csv(DATA / "p2b_scores_with_holdout.csv", index=False)
    ok = out[out["error"].isna()].dropna(subset=["composite_score"])
    print(f"scored {len(jobs)} tickers x {len(HOLDOUT_DATES)} holdout dates in {time.time() - t0:.0f}s; usable composite scores: {len(ok)} "
          f"({ok.groupby('date').size().to_dict()})")


def max_dd(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def evaluate():
    if RESULT.exists():
        sys.exit(f"The holdout was already evaluated ({RESULT.name}). It is a one-shot test and cannot be re-run.")
    mp = DATA / "frozen_model_2025-06-30.joblib"
    sha = hashlib.sha256(mp.read_bytes()).hexdigest()
    assert sha == FROZEN_SHA, f"frozen model file changed! {sha}"
    model = joblib.load(mp)
    fi.DEV_END = pd.Timestamp("2026-06-30")                       # the ONLY place the development limit is lifted, for this test
    m.SCORES_FILE = "p2b_scores_with_holdout.csv"
    P, _ = m.build_panel()
    P = P.reset_index(drop=True)
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    X = m.prep(P)
    missing = [c for c in model["columns"] if c not in X.columns]
    assert not missing, f"feature columns missing: {missing}"
    X = X[model["columns"]]
    H = P[P.date.isin(HOLDOUT_DATES)].copy()
    assert H.date.nunique() == 4, H.date.unique()
    H["pred_hgb"] = model["hgb"].predict(X.loc[H.index])
    H["pred_ridge"] = model["ridge"].predict(X.loc[H.index])
    out = {"frozen_sha256": sha, "dates": [str(d.date()) for d in HOLDOUT_DATES]}
    for name in ("pred_hgb", "pred_ridge"):
        held_prev, rows = set(), []
        for t in HOLDOUT_DATES:
            g = H[(H.date == t) & H.fwd.notna() & H[name].notna()]
            top = g.nlargest(max(10, len(g) // 10), name)
            held = set(top.ticker)
            turnover = 1.0 if not held_prev else 1 - len(held & held_prev) / len(held)
            ic = g[name].corr(g.fwd, method="spearman")
            rows.append({"date": str(t.date()), "n_scored": len(g), "n_top": len(top), "gross_top": top.fwd.mean(), "bench": g.fwd.mean(),
                         "turnover": turnover, "ic_1q": ic,
                         "net_top": top.fwd.mean() - turnover * 2 * COST_BPS / 1e4})
            held_prev = held
        r = pd.DataFrame(rows)
        r["net_excess"] = r.net_top - r.bench
        r["gross_excess"] = r.gross_top - r.bench
        wins = int((r.net_excess > 0).sum())
        mean_net = float(r.net_excess.mean())
        verdict = ("CONFIRMED (directionally, not proven)" if mean_net >= 0.005 and wins >= 3 else
                   "NOT CONFIRMED" if mean_net <= 0 or wins <= 1 else "INCONCLUSIVE")
        # descriptive: the one date with a full 4-quarter return
        t0 = HOLDOUT_DATES[0]; t4 = pd.Timestamp("2026-09-30")
        g0 = H[H.date == t0].copy()
        g0["fwd4"] = [adj.at[t4, tk] / adj.at[t0, tk] - 1 if tk in adj.columns else np.nan for tk in g0.ticker]
        g0 = g0.dropna(subset=["fwd4", name])
        top0 = g0.nlargest(max(10, len(g0) // 10), name)
        out[name] = {"verdict": verdict, "mean_net_excess_per_quarter": mean_net, "quarters_beating_benchmark": wins,
                     "annualised_net_excess": mean_net * 4, "mean_gross_excess_per_quarter": float(r.gross_excess.mean()),
                     "mean_ic_1q": float(r.ic_1q.mean()), "ic_4q_single_date": float(g0[name].corr(g0.fwd4, method="spearman")),
                     "top_decile_4q_excess_single_date": float(top0.fwd4.mean() - g0.fwd4.mean()),
                     "net_return_total": float((1 + r.net_top).prod() - 1), "bench_return_total": float((1 + r.bench).prod() - 1),
                     "per_quarter": rows}
        print(f"\n=== {name}: top-decile long-only, 4 holdout quarters, {COST_BPS} bps one-way ===")
        print(r[["date", "n_scored", "n_top", "gross_top", "bench", "turnover", "net_excess", "ic_1q"]].round(4).to_string(index=False))
        print(f"mean net excess/quarter {mean_net * 100:+.2f} pts (annualised {mean_net * 400:+.1f}); beats benchmark in {wins}/4 quarters; "
              f"total net {out[name]['net_return_total'] * 100:+.1f}% vs benchmark {out[name]['bench_return_total'] * 100:+.1f}%")
        print(f"mean 1q IC {r.ic_1q.mean():+.4f}; 4q IC on 2025-09-30 {out[name]['ic_4q_single_date']:+.4f}; top-decile 4q excess {out[name]['top_decile_4q_excess_single_date'] * 100:+.1f} pts")
        print(f"PRE-REGISTERED VERDICT ({name}): {verdict}")
    RESULT.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    {"score": score, "eval": evaluate}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
