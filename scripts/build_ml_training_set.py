"""
build_ml_training_set.py

Builds a labeled training set for the (display-only, not yet folded
into the recommendation composite) ML valuation classifier -- see
app/valuation/ml_valuation_classifier.py.

Reuses phase2_backtest.py's already-validated point-in-time
infrastructure (no-look-ahead financials/prices/beta as of the as-of
date -- filing-lag-filtered statements, price as of the as-of date,
trailing beta) instead of re-deriving it. This directly avoids the
exact defect the separate DCF Valuation Engine project's own
"historical demo" training data had: that project computed valuation
FEATURES from TODAY's fundamentals but graded them against a price
from 6 months ago -- a real look-ahead bias its own README and code
comments flag but that still ended up baked into its headline
accuracy claim. Here, both the feature vector and the price are
genuinely as of the same historical as-of date; only the realized
label (computed from the return between then and today) looks
forward, which is the correct and unavoidable way to generate ground
truth for any forward-return-based label.

Originally a single hardcoded as-of date (12 months ago) against the
~100-ticker curated universe -- 39-49 usable rows, nowhere near enough
for a classifier to learn real structure. Now takes the same
`as_of_months_ago` / `--universe` / `--workers` arguments
phase2_backtest.py already does, so it can be pointed at the full
1,002-ticker broad universe across several non-overlapping historical
windows and run_ml_training_sweep.py (this directory) can combine the
per-window outputs into one large training set.

Label definition matches the accuracy definition already used
throughout this project (phase2_backtest.py's BUY/SELL_THRESHOLD):
UNDERVALUED if realized return > +5%, OVERVALUED if < -5%, else
FAIRLY VALUED -- the same +-5% band, not a different one invented for
this script.

Output: ml_training_set_{universe_tag}_asof{N}mo.csv in this directory
(feature columns + realized_label + bookkeeping columns for
traceability) -- one file per (universe, as-of date) combination, so
concurrent/repeated runs across different windows never clobber each
other. combine_ml_training_sets.py concatenates them into the single
ml_training_set.csv app/valuation/ml_valuation_classifier.py reads.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf
from datetime import datetime

import phase2_backtest as bt
from app.valuation.ml_features import FEATURE_COLUMNS

REALIZED_MARGIN_PCT = 5.0


def realized_label(realized_return_pct: float) -> str:
    if realized_return_pct > REALIZED_MARGIN_PCT:
        return "UNDERVALUED"
    if realized_return_pct < -REALIZED_MARGIN_PCT:
        return "OVERVALUED"
    return "FAIRLY VALUED"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Builds one window's worth of labeled ML training rows."
    )
    parser.add_argument("as_of_months_ago", nargs="?", type=int, default=bt.BACKTEST_MONTHS_AGO)
    parser.add_argument(
        "--universe", type=str, default=None,
        help="Path to a {ticker: category} JSON file (e.g. scripts/ticker_universe.json) "
             "to run against instead of the hand-curated TICKERS dict.",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Concurrent worker threads (default: %(default)s) -- same rate-limit-conscious "
             "default as phase2_backtest.py, which this script was previously missing entirely "
             "(a sequential for-loop), making a broad-universe run impractically slow.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    as_of_months_ago = args.as_of_months_ago

    if args.universe:
        with open(args.universe) as f:
            tickers = json.load(f)
        universe_tag = Path(args.universe).stem
    else:
        tickers = bt.TICKERS
        universe_tag = "curated"

    today_date = pd.Timestamp(datetime.utcnow().date())
    as_of_date = today_date - pd.Timedelta(days=as_of_months_ago * 30)

    output_path = str(
        Path(__file__).resolve().parent
        / f"ml_training_set_{universe_tag}_asof{as_of_months_ago}mo.csv"
    )

    print(f"Universe: {universe_tag} ({len(tickers)} tickers)   Workers: {args.workers}", file=sys.stderr)
    print(f"As-of date: {as_of_date.date()}   Today: {today_date.date()}", file=sys.stderr)

    # 10y, matching phase2_backtest.py's own reasoning: a trailing beta
    # window computed as of a date already far in the past needs
    # benchmark history reaching back further still.
    market_history = bt._tz_naive(yf.Ticker(bt.MARKET_BENCHMARK).history(period="10y"))
    # Point-in-time risk-free rate + benchmark/sector/rate cutoff source
    # for run_one -- this call site was previously missing both fixes
    # phase2_backtest.py's own main() applies (see that file's
    # ctx.risk_free_rate_override and ctx.point_in_time_cutoff comments),
    # which meant every row in this training set was silently leaking
    # today's Treasury yield and today's benchmark/sector levels into a
    # historical valuation.
    tnx_history = bt._tz_naive(yf.Ticker("^TNX").history(period="10y"))

    rows = []
    skipped_no_dcf = 0
    errored = 0
    completed = 0

    def _run(ticker, category):
        return bt.run_one(ticker, category, as_of_date, today_date, market_history, tnx_history)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run, ticker, category): (ticker, category)
            for ticker, category in tickers.items()
        }
        for future in as_completed(futures):
            ticker, category = futures[future]
            completed += 1
            print(f"[{completed}/{len(tickers)}] {ticker} ({category})...", file=sys.stderr)
            try:
                result = future.result()
            except Exception as e:
                print(f"  [skip] {ticker}: {e}", file=sys.stderr)
                errored += 1
                continue

            features = result.get("ml_features")
            if features is None:
                skipped_no_dcf += 1
                continue

            rows.append({
                "ticker": ticker,
                "category": category,
                "as_of_date": result["as_of_date"],
                **features,
                "realized_return_pct": result["realized_return_pct"],
                "realized_label": realized_label(result["realized_return_pct"]),
            })

    # A fully rate-limited run (every ticker errored, rows=[]) would
    # otherwise write pd.DataFrame([]) -- a column-less frame that
    # to_csv serializes as a genuinely empty (0-byte) file,
    # unreadable by pd.read_csv (EmptyDataError, confirmed against a
    # real rate-limited run). Explicit columns keep the file a valid,
    # header-only CSV instead -- combine_ml_training_sets.py can still
    # skip it (0 rows), just without crashing on it.
    columns = ["ticker", "category", "as_of_date", *FEATURE_COLUMNS, "realized_return_pct", "realized_label"]
    df = pd.DataFrame(rows, columns=columns)
    before_dropna = len(df)
    if not df.empty:
        df = df.dropna(subset=FEATURE_COLUMNS + ["realized_label"])

    df.to_csv(output_path, index=False)

    print(file=sys.stderr)
    print(f"Universe: {len(tickers)}   Errored: {errored}   "
          f"No DCF (skipped): {skipped_no_dcf}   Usable rows: {before_dropna}   "
          f"After dropping incomplete features: {len(df)}", file=sys.stderr)
    if not df.empty:
        print(f"Label distribution:\n{df['realized_label'].value_counts()}", file=sys.stderr)
    print(f"\nSaved -> {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
