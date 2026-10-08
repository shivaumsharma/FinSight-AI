"""
data_quality_ic.py -- RESEARCH SANDBOX. Pre-registered in SPRINT_TRACKER.md ("FIX 1").

Does the learned 4-quarter model rank worse where its inputs are incomplete? Per stock-date, completeness = all 16 base
features present before filling. Mean rank IC (4-quarter) of each bucket per date, and the paired difference.
Run: python research/data_quality_ic.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import tool_ablation as t  # noqa: E402


def main():
    P, end_of = t.build()
    preds, tests = t.walk_forward(P, end_of, t.BASE_FEATS)
    complete = P[t.BASE_FEATS].notna().all(axis=1)
    print(f"complete share of stock-dates: {complete.mean():.1%}")
    rows = []
    for T in tests:
        d = P[(P.date == T) & P.fwd4.notna()]
        for k in ("hgb", "ridge"):
            row = {"date": T, "model": k}
            for name, mask in (("complete", complete), ("incomplete", ~complete)):
                g = d[mask.loc[d.index]]
                row[name] = preds[k].loc[g.index].corr(g.fwd4, method="spearman") if len(g) >= 25 else np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    for k in ("hgb", "ridge"):
        x = df[df.model == k].dropna(subset=["complete", "incomplete"])
        diff = x.complete - x.incomplete
        print(f"{k}: complete IC {x.complete.mean():+.4f} (NW t {t.nw_t(x.complete):.2f}) | incomplete IC {x.incomplete.mean():+.4f} "
              f"(NW t {t.nw_t(x.incomplete):.2f}) | complete - incomplete {diff.mean():+.4f} (NW t {t.nw_t(diff):.2f}), {len(x)} dates")


if __name__ == "__main__":
    main()
