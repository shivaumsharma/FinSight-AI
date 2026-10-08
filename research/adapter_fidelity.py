"""
adapter_fidelity.py -- RESEARCH SANDBOX. Does the EDGAR adapter reproduce the production model?
Scores the same tickers on the same date twice -- once from yfinance statements (what
production uses), once from EDGAR as-filed statements via research/pit_composite.py -- and
compares composite_score, dcf_score and DCF upside.

Run: python research/adapter_fidelity.py
"""
import sys, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import pit_composite as pc  # noqa: E402
from scripts.phase2_backtest import _fetch_raw_ticker_data, _score_ticker_at_date  # noqa: E402

AS_OF = pd.Timestamp("2025-06-30")
N = 80


def main():
    jobs = pc.prepare()
    rng = np.random.RandomState(1)
    jobs = [jobs[i] for i in rng.choice(len(jobs), size=N, replace=False)]
    market = pc.market_series()
    pc._init_worker(market)

    def yf_score(job):
        t, cat = job[0], job[1]
        try:
            raw = _fetch_raw_ticker_data(t)
            r = _score_ticker_at_date(t, cat, raw, AS_OF, AS_OF, market["^GSPC"], market["^TNX"])
            return t, r["composite_score"], r["dcf_score"], r["upside_pct"]
        except Exception:
            return t, None, None, None

    with ThreadPoolExecutor(6) as ex:
        yf_rows = list(ex.map(yf_score, jobs))
    yf = pd.DataFrame(yf_rows, columns=["ticker", "yf_comp", "yf_dcf", "yf_upside"]).set_index("ticker")
    ed = pd.DataFrame([r for j in jobs for r in pc.score_ticker((j, [AS_OF]))])
    ed = ed.set_index("ticker")[["composite_score", "dcf_score", "upside_pct", "error"]]
    ed.columns = ["ed_comp", "ed_dcf", "ed_upside", "error"]
    both = yf.join(ed)
    print(f"tickers {len(both)} | yfinance scored {both.yf_comp.notna().sum()} | EDGAR scored {both.ed_comp.notna().sum()}")
    ok = both.dropna(subset=["yf_comp", "ed_comp"])
    print(f"both scored: {len(ok)}")
    for a, b, lab in (("yf_comp", "ed_comp", "composite"), ("yf_dcf", "ed_dcf", "dcf_score"), ("yf_upside", "ed_upside", "dcf upside %")):
        x = ok.dropna(subset=[a, b])
        print(f"{lab}: Spearman {x[a].corr(x[b], method='spearman'):.2f}, median |diff| {(x[a]-x[b]).abs().median():.1f}, n={len(x)}")
    ok["diff"] = (ok.yf_comp - ok.ed_comp).abs()
    print("\nlargest composite disagreements:")
    print(ok.sort_values("diff", ascending=False).head(10).round(1).to_string())
    both.to_csv(Path(__file__).resolve().parent / "data" / "p2b_fidelity.csv")


if __name__ == "__main__":
    main()
