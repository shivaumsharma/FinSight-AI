"""
risk_range_calibration.py -- RESEARCH SANDBOX. Does the 10th-90th percentile range from app/analysis/risk_range.py
contain the realised price ~80% of the time? (Pre-registered in SPRINT_TRACKER.md, "Fix 2".)

Every quarter-end date, every ticker with >= 252 days of history: range from trailing volatility, then compare with
the actual price 63 and 252 trading days later. Reports coverage (target 80%), and the share of misses above/below.
Run: python research/risk_range_calibration.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.analysis.risk_range import HORIZONS, WINDOW, Z_80  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def main():
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    rets = adj.pct_change()
    vol = rets.rolling(WINDOW, min_periods=120).std(ddof=1) * np.sqrt(252)
    idx = adj.index.to_series()
    dates = sorted(idx.groupby(idx.index.to_period("Q")).max())
    out = {}
    for name, days in HORIZONS.items():
        rows = []
        for d in dates:
            pos = adj.index.get_loc(d)
            if pos + days >= len(adj):
                continue
            future = adj.iloc[pos + days]
            sd = vol.loc[d] * np.sqrt(days / 252)
            move = np.log(future / adj.loc[d])
            ok = sd.notna() & move.notna() & (sd > 0)
            rows.append(pd.DataFrame({"date": d, "inside": (move[ok].abs() <= Z_80 * sd[ok]),
                                      "above": move[ok] > Z_80 * sd[ok], "below": move[ok] < -Z_80 * sd[ok]}))
        df = pd.concat(rows)
        by_date = df.groupby("date").inside.mean()
        out[name] = {"coverage_pct": 100 * df.inside.mean(), "above_pct": 100 * df.above.mean(), "below_pct": 100 * df.below.mean(),
                     "worst_date_coverage_pct": 100 * by_date.min(), "best_date_coverage_pct": 100 * by_date.max(),
                     "n": len(df), "n_dates": by_date.size}
    print(pd.DataFrame(out).T.round(1).to_string())


if __name__ == "__main__":
    main()
