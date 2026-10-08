"""
pit_stability.py -- RESEARCH SANDBOX. G3 of SPRINT_TRACKER.md: rerun the EDGAR-fed production scorer and
capture the DCF's own uncertainty outputs (Monte Carlo + WACC x terminal-growth grid) next to every score.

Run: python research/pit_stability.py run      (about 45 minutes on 16 cores; writes research/data/g3_scores.csv)
"""
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import pit_composite as pc  # noqa: E402

DATA = pc.DATA
_CAPTURE = {}


def _init(market):
    pc._init_worker(market)
    import scripts.phase2_backtest as sp

    original = sp.derive_recommendation

    def capture(valuation_results, *a, **k):
        _CAPTURE["vr"] = valuation_results
        return original(valuation_results, *a, **k)

    sp.derive_recommendation = capture


def stability(vr):
    out = {"s1": np.nan, "s2": np.nan, "s3": np.nan, "mc_rel_width": np.nan}
    if not vr:
        return out
    price, upside = vr.get("current_price"), vr.get("upside_percent")
    mc = vr.get("monte_carlo")
    if mc and price:
        p = mc["prob_undervalued"]
        out["s3"] = p
        if upside is not None and not (isinstance(upside, float) and np.isnan(upside)):
            out["s1"] = p if upside > 0 else 1 - p
        out["mc_rel_width"] = (mc["p75"] - mc["p25"]) / price
    grid = vr.get("sensitivity_analysis")
    if grid is not None and price and upside is not None and not (isinstance(upside, float) and np.isnan(upside)):
        vals = pd.to_numeric(pd.DataFrame(grid).stack(), errors="coerce").dropna()
        if len(vals):
            above = (vals > price).mean()
            out["s2"] = above if upside > 0 else 1 - above
    return out


def work(job_and_dates):
    from scripts.phase2_backtest import _score_ticker_at_date

    (ticker, category, ann, splits, prices), dates = job_and_dates
    market, rows = pc._STATE["market"], []
    for as_of in dates:
        if prices.index.min() > as_of - pd.Timedelta(days=300):
            continue
        try:
            raw = pc.build_raw(ann, splits, prices, as_of)
            if raw is None:
                continue
            _CAPTURE.pop("vr", None)
            res = _score_ticker_at_date(ticker, category, raw, as_of, as_of, market["^GSPC"], market["^TNX"])
        except Exception as exc:
            rows.append({"ticker": ticker, "date": as_of, "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
            continue
        rows.append({"ticker": ticker, "date": as_of, "composite_score": res["composite_score"], "dcf_score": res["dcf_score"],
                     "relative_score": res["relative_score"], "upside_pct": res["upside_pct"], "error": None,
                     **stability(_CAPTURE.get("vr"))})
    return rows


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        print(__doc__)
        return
    dates = [d for d in pc.quarter_ends() if d <= pc.DEV_END]
    jobs = pc.prepare()
    market = pc.market_series()
    t0 = time.time()
    results = []
    with Pool(16, initializer=_init, initargs=(market,)) as pool:
        for i, chunk in enumerate(pool.imap_unordered(work, [(j, dates) for j in jobs], chunksize=1), 1):
            results.append(chunk)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} tickers done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    out = pd.DataFrame([r for c in results for r in c])
    out.to_csv(DATA / "g3_scores.csv", index=False)
    ok = out[out["error"].isna()]
    print(f"{len(jobs)} tickers x {len(dates)} dates in {time.time() - t0:.0f}s -> {len(ok)} scored; "
          f"stability available for {ok['s1'].notna().sum()} (S1) / {ok['s2'].notna().sum()} (S2) / {ok['s3'].notna().sum()} (S3)")


if __name__ == "__main__":
    main()
