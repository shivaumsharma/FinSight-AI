"""
tune_ddm_weight.py

Tests whether blending the dividend discount model (DDMEngine) into
the recommendation composite actually improves accuracy -- not
assumed, measured, same discipline as tune_recommendation_config.py,
tune_ml_weight.py, and tune_momentum_weight.py.

Structurally cleaner test than tune_momentum_weight.py's: DDM_WEIGHT
never changes WHICH tickers get scored (DDMEngine.is_dividend_payer
already gates that, fixed regardless of weight) -- it only changes the
SCORE for a fixed population. That sidesteps the volume-reduction trap
momentum's test fell into (fewer Sell calls mechanically raising
accuracy in a bull-dominated sample, unrelated to genuine selectivity)
-- here, N is constant, so any accuracy change on the DDM-available
subset is attributable to the blend itself, not a shifting population.

ddm_upside_pct = (ddm_value - price_as_of) / price_as_of * 100, the
same "intrinsic value vs price" percentage DCF's own upside_pct is --
percentile-ranked against the SAME _DCF_UPSIDE_PCT_PERCENTILES table
DCF uses (not a new table built from this run's ~70 DDM-available
rows, which would be too small a sample to trust on its own -- both
upside_pct and ddm_upside_pct are literally the same kind of quantity,
so reusing the larger, already-validated reference distribution is
more defensible than a noisy new one).

Run: python scripts/tune_ddm_weight.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.baseline_scoring import score_rating
from app.reporting.report_data_builder import _dcf_score, _relative_score, _rating_from_score, _percentile_rank_score, _DCF_UPSIDE_PCT_PERCENTILES

SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW_FILES = [
    "backtest_results_curated_asof12mo_exit0mo.json",
    "backtest_results_curated_asof24mo_exit12mo.json",
]

PROD_DCF_RATIO = 0.8
PROD_RELATIVE_RATIO = 0.2

DDM_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def ddm_score(ddm_value, price_as_of):
    if ddm_value is None or not price_as_of:
        return None
    ddm_upside_pct = (ddm_value - price_as_of) / price_as_of * 100
    return _percentile_rank_score(ddm_upside_pct, _DCF_UPSIDE_PCT_PERCENTILES)


def load_window(filename: str):
    path = SCRIPT_DIR / filename
    with open(path) as f:
        rows = json.load(f)
    scoreable = []
    for r in rows:
        if r.get("dcf_score") is None or r.get("realized_return_pct") is None:
            continue
        r["ddm_score"] = ddm_score(r.get("ddm_value"), r.get("price_as_of"))
        scoreable.append(r)
    return scoreable


def composite_with_ddm(row, ddm_weight: float) -> float:
    dcf_weight = (1.0 - ddm_weight) * PROD_DCF_RATIO
    relative_weight = (1.0 - ddm_weight) * PROD_RELATIVE_RATIO
    relative_score = row.get("relative_score")
    ddm = row.get("ddm_score")

    if ddm is None:
        # No dividend-payer DDM value for this ticker -- unchanged
        # today's-production behavior, DDM_WEIGHT never applies.
        if relative_score is None:
            return row["dcf_score"]
        return PROD_DCF_RATIO * row["dcf_score"] + PROD_RELATIVE_RATIO * relative_score

    if relative_score is None:
        total = dcf_weight + ddm_weight
        return (dcf_weight / total) * row["dcf_score"] + (ddm_weight / total) * ddm
    return dcf_weight * row["dcf_score"] + relative_weight * relative_score + ddm_weight * ddm


def evaluate(rows, ddm_weight: float, ddm_only: bool):
    correct = 0
    scored = 0
    for row in rows:
        if ddm_only and row.get("ddm_score") is None:
            continue
        composite = composite_with_ddm(row, ddm_weight)
        rating = _rating_from_score(composite)
        result = score_rating(rating, row["realized_return_pct"])
        if result is None:
            continue
        scored += 1
        if result:
            correct += 1
    if scored == 0:
        return None
    return correct, scored, 100 * correct / scored


def main():
    windows = {}
    for filename in WINDOW_FILES:
        rows = load_window(filename)
        n_ddm = sum(1 for r in rows if r.get("ddm_score") is not None)
        windows[filename] = rows
        print(f"  {filename}: {len(rows)} scoreable rows, {n_ddm} with a DDM value", file=sys.stderr)

    print("\n=== Accuracy on the FULL universe (most rows unaffected -- sanity check only) ===", file=sys.stderr)
    print(f"{'DDM_WEIGHT':>11}{'Accuracy':>11}", file=sys.stderr)
    for ddm_weight in DDM_WEIGHTS:
        total_correct = total_scored = 0
        for rows in windows.values():
            outcome = evaluate(rows, ddm_weight, ddm_only=False)
            if outcome:
                c, s, _ = outcome
                total_correct += c
                total_scored += s
        acc = 100 * total_correct / total_scored if total_scored else None
        print(f"{ddm_weight:>11.2f}{acc:>10.1f}%", file=sys.stderr)

    print("\n=== Accuracy on ONLY the DDM-available subset (fixed N regardless of weight -- the real test) ===", file=sys.stderr)
    print(f"{'DDM_WEIGHT':>11}{'Accuracy':>11}{'N':>6}", file=sys.stderr)
    ddm_only_by_weight = {}
    for ddm_weight in DDM_WEIGHTS:
        total_correct = total_scored = 0
        for rows in windows.values():
            outcome = evaluate(rows, ddm_weight, ddm_only=True)
            if outcome:
                c, s, _ = outcome
                total_correct += c
                total_scored += s
        acc = 100 * total_correct / total_scored if total_scored else None
        ddm_only_by_weight[ddm_weight] = acc
        print(f"{ddm_weight:>11.2f}{acc:>10.1f}%{total_scored:>6}", file=sys.stderr)

    baseline = ddm_only_by_weight[0.0]
    best_weight = max(ddm_only_by_weight, key=lambda w: ddm_only_by_weight[w])
    print(f"\nOn the DDM-available subset: {baseline:.1f}% at weight=0 (DCF/relative only, today's config) "
          f"vs. {ddm_only_by_weight[best_weight]:.1f}% at best weight={best_weight}.", file=sys.stderr)

    near_best = [w for w, acc in ddm_only_by_weight.items() if ddm_only_by_weight[best_weight] - acc <= 1.0]
    print(f"{len(near_best)} of {len(DDM_WEIGHTS)} tested weights within 1 point of the best "
          f"({'plateau' if len(near_best) >= 4 else 'narrow spike -- treat with caution'}).", file=sys.stderr)
    print("\nNote: small sample (see N above) -- a real signal should still be treated as directional "
          "evidence, not a settled result, until more dividend-payer point-in-time data exists.", file=sys.stderr)


if __name__ == "__main__":
    main()
