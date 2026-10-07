"""
freeze_model.py -- RESEARCH SANDBOX. Freeze the learned 4-quarter model exactly as the walk-forward used it at the end of the
development period, so any later test (the sealed holdout, the live record) is a pure application of a LOCKED model.

Training data = every row whose 4-quarter label was fully realised on or before 2025-06-30 (the same rule as the last
walk-forward retrain). Models and settings are the A1b ones, unchanged: HistGradientBoostingRegressor(max_depth=3, learning_rate
0.05, max_iter 200, min_samples_leaf 200, l2 1.0, random_state 0) and Ridge(alpha 10), target = per-date percentile rank of the
next-4-quarter return, features = the 16 base features (rank-normalised per date) + sector dummies.

Outputs (gitignored): research/data/frozen_model_2025-06-30.joblib and a manifest with the file's SHA-256, written below and
copied into SPRINT_TRACKER.md so the freeze is auditable. Also verifies the frozen model reproduces the walk-forward
predictions for the freeze date.

Run: python research/freeze_model.py
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import ml_cross_section as m  # noqa: E402

DATA = m.DATA
FREEZE = pd.Timestamp("2025-06-30")
H = 4


def main():
    P, _ = m.build_panel()
    P = P.reset_index(drop=True)
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    idx = adj.index.to_series()
    qe = [d for d in sorted(idx.groupby(idx.index.to_period("Q")).max()) if d >= m.fi.FIRST_DATE]
    end_of = {d: qe[i + H] for i, d in enumerate(qe) if i + H < len(qe)}
    P["fwd4"] = [adj.at[end_of[d], t] / adj.at[d, t] - 1 if d in end_of and t in adj.columns else np.nan for d, t in zip(P.date, P.ticker)]
    P["y"] = P.groupby("date")["fwd4"].rank(pct=True)
    X = m.prep(P)
    tr = P.date.map(lambda d: d in end_of and end_of[d] <= FREEZE) & P.y.notna()
    hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                        l2_regularization=1.0, random_state=0).fit(X[tr], P.loc[tr, "y"])
    ridge = Ridge(alpha=10).fit(X[tr], P.loc[tr, "y"])
    out = DATA / "frozen_model_2025-06-30.joblib"
    joblib.dump({"hgb": hgb, "ridge": ridge, "columns": list(X.columns), "feats": m.FEATS, "freeze_date": str(FREEZE.date()),
                 "train_rows": int(tr.sum()), "train_dates": [str(P.loc[tr, "date"].min().date()), str(P.loc[tr, "date"].max().date())]}, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    # reproducibility check: the walk-forward predictions at the freeze date came from the same training rule
    te = P.date == FREEZE
    wf = pd.read_parquet(DATA / "a1b_predictions.parquet")
    wf = wf[wf.date == FREEZE].set_index("ticker")
    mine = pd.Series(hgb.predict(X[te]), index=P.loc[te, "ticker"].values)
    j = pd.concat([mine.rename("frozen"), wf["pred_hgb"].rename("walkforward")], axis=1).dropna()
    manifest = {"file": out.name, "sha256": sha, "train_rows": int(tr.sum()), "train_date_range": [str(P.loc[tr, "date"].min().date()), str(P.loc[tr, "date"].max().date())],
                "n_features": len(X.columns), "reproduces_walkforward_at_freeze_date": {"names": len(j), "correlation": round(float(j.corr().iloc[0, 1]), 6)}}
    (DATA / "frozen_model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
