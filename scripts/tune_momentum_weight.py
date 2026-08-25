"""
tune_momentum_weight.py

Tests the "don't sell into strength" fix: does gating Sell calls with
6-month price momentum actually improve accuracy, specifically Sell
precision -- not assumed, measured, same discipline as
tune_recommendation_config.py's DCF_WEIGHT/RELATIVE_WEIGHT grid search
and tune_ml_weight.py's ML_WEIGHT test.

Motivation (EVALUATION.md section 1/6): the model's worst Sell calls
this session were semiconductor names with strong momentum the
composite score never saw -- DCF said "overvalued," the stocks kept
running, the model had no mechanism to notice the trend it was selling
into.

Asymmetric by design: momentum only ever pulls the composite score UP
(toward Buy/away from Sell), never down. Buy precision is already the
model's stronger signal (54-65% across every window tested) and
doesn't need help; this targets the diagnosed Sell weakness
specifically, without also making the model chase momentum on the Buy
side (a different, riskier bet this fix isn't trying to make).

momentum_score is percentile-rank normalized against real historical
momentum_6m values (n=1,699, from the Phase 2 training-set sweep,
winsorized to [-80, 150] the same way the DCF/relative-valuation
tables are) -- same "different units need normalizing before blending
at a fixed weight" discipline report_data_builder.py's own comment
documents, not raw-percentage blending.

No retraining needed (unlike tune_ml_weight.py) -- momentum_6m is a
raw, already-computed feature, not a fitted model's prediction, so
there's no in-sample-leakage risk to guard against here. Directly
reuses report_data_builder.py's own _dcf_score/_relative_score/
_percentile_rank_score functions.

Run: python scripts/tune_momentum_weight.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.analysis.baseline_scoring import score_rating
from app.reporting.report_data_builder import _dcf_score, _relative_score, _rating_from_score, _percentile_rank_score

SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW_FILES = [
    "ml_training_set_ticker_universe_asof12mo.csv",
    "ml_training_set_ticker_universe_asof18mo.csv",
    "ml_training_set_ticker_universe_asof24mo.csv",
    "ml_training_set_ticker_universe_asof30mo.csv",
]

PROD_DCF_RATIO = 0.8
PROD_RELATIVE_RATIO = 0.2

MOMENTUM_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Winsorized [-80, 150] the same way DCF's own table is (see that
# table's own comment) -- one 280%-momentum outlier would otherwise
# stretch the whole top of the scale. Built from all 4 Phase 2 windows
# combined (n=1,699), percentiles 0,5,...,100.
_MOMENTUM_6M_PERCENTILES = [
    -59.3, -24.6, -17.5, -12.8, -9.7, -7.4, -4.4, -2.3, -0.4, 1.6,
    3.5, 5.7, 8.3, 10.6, 13.2, 16.2, 20.0, 23.9, 30.0, 38.9, 150.0,
]


def momentum_score(momentum_6m):
    if pd.isna(momentum_6m):
        return None
    # Asymmetric: only the bullish half of the percentile rank ever
    # applies (a below-median momentum reading contributes 0, never a
    # negative pull -- this fix targets Sell precision specifically,
    # not "also chase momentum on Buy calls").
    return max(0.0, _percentile_rank_score(momentum_6m, _MOMENTUM_6M_PERCENTILES))


def load_window(filename: str) -> pd.DataFrame:
    df = pd.read_csv(SCRIPT_DIR / filename)
    df["upside_percent"] = (df["dcf_over_price"] - 1.0) * 100.0
    df["dcf_score"] = df["upside_percent"].apply(_dcf_score)
    df["relative_score"] = df["relative_vs_history_pct"].apply(
        lambda v: _relative_score({"vs_history_pct": v}) if pd.notna(v) else None
    )
    df["momentum_score"] = df["momentum_6m"].apply(momentum_score)
    return df


def composite_with_momentum(row, momentum_weight: float) -> float:
    dcf_weight = (1.0 - momentum_weight) * PROD_DCF_RATIO
    relative_weight = (1.0 - momentum_weight) * PROD_RELATIVE_RATIO
    relative_score = row["relative_score"]
    mom = row["momentum_score"] if pd.notna(row["momentum_score"]) else 0.0

    if pd.isna(relative_score):
        total = dcf_weight + momentum_weight
        if total == 0:
            return row["dcf_score"]
        return (dcf_weight / total) * row["dcf_score"] + (momentum_weight / total) * mom
    return dcf_weight * row["dcf_score"] + relative_weight * relative_score + momentum_weight * mom


def evaluate(df: pd.DataFrame, momentum_weight: float):
    correct = 0
    scored = 0
    sell_correct = 0
    sell_scored = 0
    for _, row in df.iterrows():
        composite = composite_with_momentum(row, momentum_weight)
        rating = _rating_from_score(composite)
        result = score_rating(rating, row["realized_return_pct"])
        if result is None:
            continue
        scored += 1
        if result:
            correct += 1
        if rating == "Sell":
            sell_scored += 1
            if result:
                sell_correct += 1
    if scored == 0:
        return None
    sell_precision = 100 * sell_correct / sell_scored if sell_scored else None
    return correct, scored, 100 * correct / scored, sell_correct, sell_scored, sell_precision


def main():
    windows = {}
    for filename in WINDOW_FILES:
        path = SCRIPT_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename}: not found", file=sys.stderr)
            continue
        df = load_window(filename).dropna(subset=["dcf_score", "momentum_6m"])
        if df.empty:
            print(f"  [skip] {filename}: 0 usable rows", file=sys.stderr)
            continue
        windows[filename] = df
        print(f"  {filename}: {len(df)} rows", file=sys.stderr)

    header = f"\n{'MOM_WEIGHT':>11}{'Accuracy':>11}{'Sell N':>9}{'Sell Prec':>11}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    pooled_by_weight = {}
    for momentum_weight in MOMENTUM_WEIGHTS:
        total_correct = total_scored = total_sell_correct = total_sell_scored = 0
        for df in windows.values():
            outcome = evaluate(df, momentum_weight)
            if outcome is None:
                continue
            correct, scored, _, sell_correct, sell_scored, _ = outcome
            total_correct += correct
            total_scored += scored
            total_sell_correct += sell_correct
            total_sell_scored += sell_scored
        acc = 100 * total_correct / total_scored if total_scored else None
        sell_prec = 100 * total_sell_correct / total_sell_scored if total_sell_scored else None
        pooled_by_weight[momentum_weight] = (acc, total_sell_scored, sell_prec)
        print(f"{momentum_weight:>11.2f}{acc:>10.1f}%{total_sell_scored:>9}{(f'{sell_prec:.1f}%' if sell_prec is not None else '--'):>11}", file=sys.stderr)

    best_weight = max(pooled_by_weight, key=lambda w: pooled_by_weight[w][0])
    baseline_acc = pooled_by_weight[0.0][0]
    print(f"\nBest pooled MOMENTUM_WEIGHT (by overall accuracy): {best_weight} -> "
          f"{pooled_by_weight[best_weight][0]:.1f}% (vs. {baseline_acc:.1f}% at weight=0, today's production config)", file=sys.stderr)
    print(f"Sell precision at weight=0: {pooled_by_weight[0.0][2]:.1f}% (n={pooled_by_weight[0.0][1]})   "
          f"at best weight: {pooled_by_weight[best_weight][2]:.1f}% (n={pooled_by_weight[best_weight][1]})", file=sys.stderr)

    near_best = [w for w, (acc, _, _) in pooled_by_weight.items() if pooled_by_weight[best_weight][0] - acc <= 1.0]
    print(f"\n{len(near_best)} of {len(MOMENTUM_WEIGHTS)} tested weights are within 1 point of the best "
          f"({'looks like a robust plateau' if len(near_best) >= 4 else 'looks like a narrow spike -- treat with caution'}).")


if __name__ == "__main__":
    main()
