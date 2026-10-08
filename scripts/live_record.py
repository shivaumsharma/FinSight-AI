"""
live_record.py

Append-only weekly record of FinSight's composite scores: a forward test that
survivorship and look-ahead cannot contaminate.

    python scripts/live_record.py snapshot [--limit N] [--workers 4]
    python scripts/live_record.py evaluate [--horizon-days 63]

Rows go to live_record/snapshots.csv (a same-day re-run never overwrites).
`evaluate` joins matured snapshots to realised returns and prints no t-statistic
until 8 snapshot dates have matured. Commit the CSV to keep the record.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECORD_PATH = ROOT / "live_record" / "snapshots.csv"
UNIVERSE_PATH = ROOT / "scripts" / "ticker_universe.json"
COLUMNS = ["snapshot_date", "ticker", "category", "composite_score", "dcf_score", "relative_score", "upside_pct",
           "recommendation", "price"]
MIN_DATES_FOR_T = 8  # below this a t-stat is noise dressed as a number


# ----------------------------------------------------------------------------- pure helpers (unit-tested)

def append_snapshot(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add `new` rows to the record. A (snapshot_date, ticker) pair already stored is never overwritten:
    the record is append-only, so re-running a day cannot rewrite history."""
    if existing is None or existing.empty:
        combined = new.copy()
    else:
        keys = set(zip(existing["snapshot_date"], existing["ticker"]))
        fresh = new[[(d, t) not in keys for d, t in zip(new["snapshot_date"], new["ticker"])]]
        combined = pd.concat([existing, fresh], ignore_index=True)
    return combined[COLUMNS].sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)


def forward_return(prices: pd.Series, snapshot_date, horizon_days: int):
    """Total-return-style return from the first close ON/AFTER the snapshot date to `horizon_days` calendar days
    later (first close on/after that). None if the horizon has not matured yet or prices are missing."""
    prices = prices.dropna()
    if prices.empty:
        return None
    start_date = pd.Timestamp(snapshot_date)
    end_date = start_date + pd.Timedelta(days=horizon_days)
    if prices.index.max() < end_date:
        return None
    p0 = prices[prices.index >= start_date]
    p1 = prices[prices.index >= end_date]
    if p0.empty or p1.empty:
        return None
    return float(p1.iloc[0] / p0.iloc[0] - 1)


def ic_summary(ic_by_date: pd.Series) -> dict:
    s = ic_by_date.dropna()
    n = len(s)
    out = {"dates": n, "mean_ic": float(s.mean()) if n else None, "t": None,
           "note": None}
    if n >= MIN_DATES_FOR_T and s.std(ddof=1) > 0:
        out["t"] = float(s.mean() / (s.std(ddof=1) / np.sqrt(n)))
    else:
        out["note"] = f"only {n} matured snapshot date(s); need {MIN_DATES_FOR_T} before a t-statistic means anything"
    return out


# ----------------------------------------------------------------------------- IO

def load_record() -> pd.DataFrame:
    if not RECORD_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(RECORD_PATH)


def _score_one(args):
    ticker, category, market, tnx, today = args
    from scripts.phase2_backtest import _fetch_raw_ticker_data, _price_on_or_before, _score_ticker_at_date

    try:
        raw = _fetch_raw_ticker_data(ticker)
        res = _score_ticker_at_date(ticker, category, raw, today, today, market, tnx)
        return {"snapshot_date": today.date().isoformat(), "ticker": ticker, "category": category,
                "composite_score": res["composite_score"], "dcf_score": res["dcf_score"],
                "relative_score": res["relative_score"], "upside_pct": res["upside_pct"],
                "recommendation": res["recommendation"], "price": _price_on_or_before(raw["price_history"], today)}
    except Exception:
        return None  # an unscoreable ticker simply has no row that week


def snapshot(limit=None, workers=4):
    import yfinance as yf

    universe = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    items = list(universe.items())[:limit]
    today = pd.Timestamp.today().normalize()

    def hist(sym):
        h = yf.Ticker(sym).history(period="2y")
        h.index = h.index.tz_localize(None)
        return h

    market, tnx = hist("^GSPC"), hist("^TNX")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(_score_one, [(t, c, market, tnx, today) for t, c in items]) if r]
    new = pd.DataFrame(rows, columns=COLUMNS)
    merged = append_snapshot(load_record(), new)
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(RECORD_PATH, index=False)
    print(f"{today.date()}: scored {len(new)} of {len(items)} tickers; record now holds {len(merged)} rows "
          f"over {merged['snapshot_date'].nunique()} snapshot date(s)")


def evaluate(horizon_days=63):
    import yfinance as yf

    rec = load_record()
    if rec.empty:
        print("no snapshots recorded yet -- run `snapshot` first")
        return
    cutoff = (pd.Timestamp.today().normalize() - pd.Timedelta(days=horizon_days)).date().isoformat()
    rec = rec[rec["snapshot_date"] <= cutoff]            # only snapshots old enough to have matured; no pointless downloads
    if rec.empty:
        print(f"horizon {horizon_days}d | no snapshot is {horizon_days} days old yet")
        print(ic_summary(pd.Series(dtype=float))["note"])
        return
    tickers = sorted(rec["ticker"].unique())
    px = yf.download(tickers, start=str(rec["snapshot_date"].min()), auto_adjust=True, progress=False)["Close"]
    px.index = px.index.tz_localize(None) if px.index.tz is not None else px.index
    ics, spreads = {}, {}
    for date, g in rec.groupby("snapshot_date"):
        fwd = pd.Series({t: forward_return(px[t], date, horizon_days) for t in g["ticker"] if t in px.columns}, dtype=float)
        j = g.set_index("ticker").join(fwd.rename("fwd")).dropna(subset=["composite_score", "fwd"])
        if len(j) >= 100:
            ics[date] = j["composite_score"].corr(j["fwd"], method="spearman")
            q = pd.qcut(j["composite_score"].rank(method="first"), 5, labels=False)
            spreads[date] = j["fwd"][q == 4].mean() - j["fwd"][q == 0].mean()
    s = ic_summary(pd.Series(ics, dtype=float))
    print(f"horizon {horizon_days}d | snapshot dates stored: {rec['snapshot_date'].nunique()} | matured with >=100 names: {s['dates']}")
    if s["mean_ic"] is not None:
        print(f"mean rank IC {s['mean_ic']:+.4f}" + (f" | t {s['t']:+.2f}" if s["t"] is not None else ""))
        if spreads:
            print(f"mean top-minus-bottom quintile return {np.mean(list(spreads.values())) * 100:+.2f}%")
    if s["note"]:
        print(s["note"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("snapshot")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--workers", type=int, default=4)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--horizon-days", type=int, default=63)
    args = ap.parse_args()
    if args.cmd == "snapshot":
        snapshot(args.limit, args.workers)
    else:
        evaluate(args.horizon_days)


if __name__ == "__main__":
    main()
