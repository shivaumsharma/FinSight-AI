"""
build_rlvr_dataset.py

Builds the training/held-out prompt datasets for RLVR (GRPO) direction-
prediction training -- see app/training/rlvr_context.py, rlvr_prompt.py,
rlvr_reward.py, and the project plan doc for the full design.

Reuses phase2_backtest.py's TICKERS/MARKET_BENCHMARK and
rlvr_context.build_point_in_time_context() (itself built on
phase2_backtest.py's own no-look-ahead helpers) -- same US-only universe
and point-in-time methodology as scripts/build_ml_training_set.py, for
the same reason that script gives (FinSight's RAG/sentiment layer is
SEC-EDGAR-based, so this doesn't extend to non-US filers).

Held-out split is on TWO axes, not one, so the final "did RLVR help"
claim (scripts/evaluate_rlvr_model.py) can't be overfit to either:
  1. Ticker holdout -- a fixed, seeded subset of tickers never appears
     in the training set at all, regardless of as-of-date.
  2. Window holdout -- the most recent as-of-date window is held out
     across ALL tickers (including training tickers), so even a
     training-set company's most recent snapshot is unseen.
Anything touching either holdout axis goes to the held-out file; only
examples that are both a training ticker AND a training window go to
the training file.

Output: scripts/rlvr_dataset_train.jsonl, scripts/rlvr_dataset_heldout.jsonl
Each line: {ticker, category, as_of_date, prompt, realized_return_pct}
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import phase2_backtest as bt
from app.analysis.financial_analysis import FinancialAnalysisBuilder
from app.training.rlvr_context import PointInTimeUnavailable, build_point_in_time_context
from app.training.rlvr_prompt import render_prompt

# Multiple non-overlapping historical snapshots per ticker instead of
# just one (build_ml_training_set.py's single 12mo-ago snapshot) --
# more training signal from the same ticker universe without needing a
# larger one. The most recent (12mo) window is the one held out below;
# 18/24mo are only ever used for training.
AS_OF_MONTHS_AGO_WINDOWS = [12, 18, 24]
HELDOUT_WINDOW_MONTHS_AGO = 12

TICKER_HOLDOUT_FRACTION = 0.2
TICKER_HOLDOUT_SEED = 20260802  # fixed, not wall-clock-derived: a reproducible split matters more than a "fresh" one here.

TRAIN_OUTPUT_PATH = str(Path(__file__).resolve().parent / "rlvr_dataset_train.jsonl")
HELDOUT_OUTPUT_PATH = str(Path(__file__).resolve().parent / "rlvr_dataset_heldout.jsonl")


def _select_holdout_tickers(tickers: list) -> set:
    rng = random.Random(TICKER_HOLDOUT_SEED)
    shuffled = sorted(tickers)  # sort first so the shuffle is deterministic regardless of dict insertion order
    rng.shuffle(shuffled)
    n_holdout = max(1, round(len(shuffled) * TICKER_HOLDOUT_FRACTION))
    return set(shuffled[:n_holdout])


def _build_example(ticker, category, as_of_date, today_date, market_history):
    ctx, realized_return_pct = build_point_in_time_context(ticker, as_of_date, today_date, market_history)

    summary_builder = FinancialAnalysisBuilder()
    financial_summary = summary_builder.build(ctx.normalized_financials)
    financial_summary.update(summary_builder.interpret(financial_summary))

    prompt = render_prompt(ticker, financial_summary, ctx.valuation_results)

    return {
        "ticker": ticker,
        "category": category,
        "as_of_date": as_of_date.date().isoformat(),
        "prompt": prompt,
        "realized_return_pct": round(realized_return_pct, 2),
    }


def main():
    today_date = pd.Timestamp(datetime.utcnow().date())
    holdout_tickers = _select_holdout_tickers(list(bt.TICKERS.keys()))

    print(f"Universe: {len(bt.TICKERS)} tickers (reused from phase2_backtest.py)", file=sys.stderr)
    print(f"Windows: {AS_OF_MONTHS_AGO_WINDOWS} months ago   Holdout window: {HELDOUT_WINDOW_MONTHS_AGO}mo",
          file=sys.stderr)
    print(f"Ticker holdout ({len(holdout_tickers)}/{len(bt.TICKERS)}): {sorted(holdout_tickers)}", file=sys.stderr)

    # 5y for the same reason phase2_backtest.py's main() uses 5y, not
    # 2y: the trailing beta window needs benchmark history reaching
    # back from the OLDEST as-of date tested here (24mo ago), not just
    # the most recent one.
    market_history = bt._tz_naive(__import__("yfinance").Ticker(bt.MARKET_BENCHMARK).history(period="5y"))

    train_rows, heldout_rows = [], []
    errored = 0
    total = len(bt.TICKERS) * len(AS_OF_MONTHS_AGO_WINDOWS)
    done = 0

    for ticker, category in bt.TICKERS.items():
        for months_ago in AS_OF_MONTHS_AGO_WINDOWS:
            done += 1
            as_of_date = today_date - pd.Timedelta(days=months_ago * 30)
            print(f"[{done}/{total}] {ticker} as-of {months_ago}mo ago...", file=sys.stderr)

            try:
                row = _build_example(ticker, category, as_of_date, today_date, market_history)
            except PointInTimeUnavailable as e:
                print(f"  [skip] {e}", file=sys.stderr)
                errored += 1
                continue
            except Exception as e:
                print(f"  [skip] {ticker} ({months_ago}mo): {e}", file=sys.stderr)
                errored += 1
                continue

            is_heldout = ticker in holdout_tickers or months_ago == HELDOUT_WINDOW_MONTHS_AGO
            (heldout_rows if is_heldout else train_rows).append(row)

    with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    with open(HELDOUT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in heldout_rows:
            f.write(json.dumps(row) + "\n")

    print(file=sys.stderr)
    print(f"Attempted: {total}   Errored/skipped: {errored}   "
          f"Train examples: {len(train_rows)}   Held-out examples: {len(heldout_rows)}", file=sys.stderr)
    print(f"Saved -> {TRAIN_OUTPUT_PATH}", file=sys.stderr)
    print(f"Saved -> {HELDOUT_OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
