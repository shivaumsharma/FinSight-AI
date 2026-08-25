"""
catalyst_proximity_analysis.py

Tests the hypothesis behind "catalyst-awareness" (lower confidence,
not a different rating, for a call made close to a known earnings
date): does accuracy actually differ for calls made near an earnings
report vs. calls made with no earnings report imminent? Tested before
building anything, same discipline as tune_momentum_weight.py and
tune_ddm_weight.py.

Reuses the curated-universe backtest results already on disk
(as_of_date + recommendation + realized_return_pct per ticker) and
enriches each row with days_to_next_earnings, computed from
yf.Ticker(ticker).get_earnings_dates() -- a real, point-in-time-safe
history of ACTUAL REPORTED earnings dates (confirmed live: AAPL
returns dated entries back to 2014, not just the next upcoming one).
Only the DATE is used, never the reported EPS/surprise on that date --
knowing a report's SCHEDULE as of a historical date is a real, roughly
knowable fact (a company's quarterly cadence is public), unlike
knowing its OUTCOME, which would be a genuine look-ahead leak.

Run: python scripts/catalyst_proximity_analysis.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import pandas as pd
import yfinance as yf

from app.analysis.baseline_scoring import score_rating

# A small, deliberate delay between per-ticker earnings-date fetches --
# this session already hit severe Yahoo Finance rate limiting once
# from a large sequential sweep with no throttling; this endpoint is
# lighter than the full financials/price-history pulls that caused
# that, but not worth risking a repeat over.
REQUEST_DELAY_SECONDS = 0.3

SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW_FILES = [
    "backtest_results_curated_asof12mo_exit0mo.json",
    "backtest_results_curated_asof24mo_exit12mo.json",
    "backtest_results_ticker_universe_asof12mo_exit0mo.json",
    "backtest_results_ticker_universe_asof24mo_exit12mo.json",
]

CLOSE_THRESHOLD_DAYS = 30

_earnings_cache = {}


def days_to_next_earnings(ticker: str, as_of_date: pd.Timestamp):
    if ticker not in _earnings_cache:
        try:
            dates = yf.Ticker(ticker).get_earnings_dates(limit=60)
            _earnings_cache[ticker] = dates.index.tz_localize(None) if dates is not None and not dates.empty else None
        except Exception:
            _earnings_cache[ticker] = None
        time.sleep(REQUEST_DELAY_SECONDS)

    dates = _earnings_cache[ticker]
    if dates is None:
        return None
    future = dates[dates >= as_of_date]
    if future.empty:
        return None
    return (future.min() - as_of_date).days


def main():
    rows = []
    for filename in WINDOW_FILES:
        path = SCRIPT_DIR / filename
        with open(path) as f:
            rows.extend(json.load(f))

    scoreable = [
        r for r in rows
        if r.get("recommendation") not in (None, "Insufficient Data")
        and r.get("realized_return_pct") is not None
    ]

    print(f"Fetching earnings-date history for {len(set(r['ticker'] for r in scoreable))} unique tickers...", file=sys.stderr)

    close_rows, far_rows, unknown = [], [], 0
    for r in scoreable:
        as_of = pd.Timestamp(r["as_of_date"])
        days = days_to_next_earnings(r["ticker"], as_of)
        r["days_to_next_earnings"] = days
        if days is None:
            unknown += 1
        elif days <= CLOSE_THRESHOLD_DAYS:
            close_rows.append(r)
        else:
            far_rows.append(r)

    def accuracy(bucket):
        if not bucket:
            return None, 0
        correct = sum(1 for r in bucket if score_rating(r["recommendation"], r["realized_return_pct"]))
        return 100 * correct / len(bucket), len(bucket)

    close_acc, close_n = accuracy(close_rows)
    far_acc, far_n = accuracy(far_rows)

    print(f"\nUnknown earnings date (excluded): {unknown}", file=sys.stderr)
    print(f"Close to earnings (<= {CLOSE_THRESHOLD_DAYS} days at as-of date): "
          f"n={close_n}  accuracy={close_acc:.1f}%" if close_acc is not None else "Close: no data", file=sys.stderr)
    print(f"Far from earnings (> {CLOSE_THRESHOLD_DAYS} days): "
          f"n={far_n}  accuracy={far_acc:.1f}%" if far_acc is not None else "Far: no data", file=sys.stderr)

    if close_acc is not None and far_acc is not None:
        print(f"\nDifference (far - close): {far_acc - close_acc:+.1f} points", file=sys.stderr)
        print(
            "If close-to-earnings calls are meaningfully LESS accurate than far-from-earnings "
            "calls, that supports flagging proximity as a real confidence signal. If the two are "
            "similar, there's no evidence for the hypothesis and it shouldn't be shipped.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
