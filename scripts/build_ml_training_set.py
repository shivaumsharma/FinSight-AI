"""
build_ml_training_set.py

Builds a labeled training set for the (display-only, not yet folded
into the recommendation composite) ML valuation classifier -- see
app/valuation/ml_valuation_classifier.py.

Reuses phase2_backtest.py's already-validated point-in-time
infrastructure (no-look-ahead financials/prices/beta as of 12 months
ago -- filing-lag-filtered statements, price as of the as-of date,
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

US-only universe: reuses phase2_backtest.TICKERS as-is. (The audited
project's universe also included ~40 Indian (.NS) tickers -- out of
scope here, since FinSight's RAG/sentiment layer is built on SEC
EDGAR, which doesn't cover Indian filers.)

Label definition matches the accuracy definition already used
throughout this project (phase2_backtest.py's BUY/SELL_THRESHOLD):
UNDERVALUED if realized return > +5%, OVERVALUED if < -5%, else
FAIRLY VALUED -- the same ±5% band, not a different one invented for
this script.

Output: ml_training_set.csv in this directory (feature columns +
realized_label + bookkeeping columns for traceability).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf
from datetime import datetime

import phase2_backtest as bt
from app.valuation.ml_features import FEATURE_COLUMNS

REALIZED_MARGIN_PCT = 5.0
OUTPUT_PATH = str(Path(__file__).resolve().parent / "ml_training_set.csv")


def realized_label(realized_return_pct: float) -> str:
    if realized_return_pct > REALIZED_MARGIN_PCT:
        return "UNDERVALUED"
    if realized_return_pct < -REALIZED_MARGIN_PCT:
        return "OVERVALUED"
    return "FAIRLY VALUED"


def main():
    today_date = pd.Timestamp(datetime.utcnow().date())
    as_of_date = today_date - pd.Timedelta(days=bt.BACKTEST_MONTHS_AGO * 30)

    print(f"As-of date: {as_of_date.date()}   Today: {today_date.date()}", file=sys.stderr)
    print(f"Universe: {len(bt.TICKERS)} tickers (reused from phase2_backtest.py)", file=sys.stderr)

    market_history = bt._tz_naive(yf.Ticker(bt.MARKET_BENCHMARK).history(period="2y"))

    rows = []
    skipped_no_dcf = 0
    errored = 0
    for ticker, category in bt.TICKERS.items():
        print(f"[{len(rows) + skipped_no_dcf + errored + 1}/{len(bt.TICKERS)}] {ticker} ({category})...",
              file=sys.stderr)
        try:
            result = bt.run_one(ticker, category, as_of_date, today_date, market_history)
        except Exception as e:
            print(f"  [skip] {ticker}: {e}", file=sys.stderr)
            errored += 1
            continue

        features = result.get("ml_features")
        if features is None:
            skipped_no_dcf += 1
            continue

        row = {
            "ticker": ticker,
            "category": category,
            "as_of_date": result["as_of_date"],
            **features,
            "realized_return_pct": result["realized_return_pct"],
            "realized_label": realized_label(result["realized_return_pct"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    before_dropna = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["realized_label"])

    df.to_csv(OUTPUT_PATH, index=False)

    print(file=sys.stderr)
    print(f"Universe: {len(bt.TICKERS)}   Errored: {errored}   "
          f"No DCF (skipped): {skipped_no_dcf}   Usable rows: {before_dropna}   "
          f"After dropping incomplete features: {len(df)}", file=sys.stderr)
    print(f"Label distribution:\n{df['realized_label'].value_counts()}", file=sys.stderr)
    print(f"\nSaved -> {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
