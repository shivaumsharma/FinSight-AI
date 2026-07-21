"""
tune_recommendation_config.py

Grid-searches BUY_THRESHOLD/SELL_THRESHOLD (the Hold band width) and
DCF_WEIGHT/RELATIVE_WEIGHT (the composite blend) against the raw
per-ticker results already captured by phase2_backtest.py
(scripts/backtest_results.json -- dcf_score, relative_score,
realized_return_pct per ticker). No network calls, no re-running the
pipeline: composite_score = w1*dcf_score + w2*relative_score and the
resulting rating are pure recomputation, so trying hundreds of
combinations here costs seconds, not the ~15 minutes a real backtest
run takes.

Caution about overfitting, stated rather than hidden: this is one
12-month window on ~50-65 scored tickers. A combination that wins this
grid search by a couple of points, on a sample this small, is not
strong evidence it's genuinely better going forward -- it can easily
be noise from this specific window (this backtest ran during a
strongly up market, see phase2_backtest.py's own market-base-rate
comparison). This script reports the full grid, not just the winner,
specifically so a human can judge whether the "best" config is a
robust plateau (many nearby combinations score similarly well) or an
isolated spike (only one narrow combination stands out, a classic
overfitting signature) before trusting it enough to change production
thresholds.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_PATH = str(Path(__file__).resolve().parent / "backtest_results_asof12mo_exit0mo.json")

# Same accuracy definition phase2_backtest.py and build_ml_training_set.py
# already use -- not a different one invented for this script.
SCORE_BUY_THRESHOLD = 5.0
SCORE_SELL_THRESHOLD = -5.0

# Grid ranges. Hold-band half-widths from tight (5, closer to the
# scoring band itself) to the current production value (15) and a bit
# wider (25), so the current config sits inside the tested range, not
# at an edge. Weights swept in 10% steps including the two extremes
# (100/0 = DCF alone, 0/100 = relative valuation alone).
BAND_WIDTHS = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0]
DCF_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SCORE_CAP = 100.0


def score_rating(rating, realized_return_pct):
    if rating is None or realized_return_pct is None:
        return None
    if rating == "Buy":
        return realized_return_pct > SCORE_BUY_THRESHOLD
    if rating == "Sell":
        return realized_return_pct < SCORE_SELL_THRESHOLD
    if rating == "Hold":
        return SCORE_SELL_THRESHOLD <= realized_return_pct <= SCORE_BUY_THRESHOLD
    return None


def rating_from_score(score, band_width):
    if score >= band_width:
        return "Buy"
    if score <= -band_width:
        return "Sell"
    return "Hold"


def load_rows(path=None):
    with open(path or RESULTS_PATH) as f:
        rows = json.load(f)
    # Only rows where DCF actually ran have a dcf_score to blend --
    # Insufficient-Data/fallback-only rows aren't part of what this
    # grid tunes (the composite blend doesn't apply to them either way).
    usable = [
        r for r in rows
        if not r.get("error")
        and r.get("dcf_score") is not None
        and r.get("realized_return_pct") is not None
    ]
    return usable


def evaluate_config(rows, band_width, dcf_weight):
    relative_weight = 1.0 - dcf_weight
    correct = 0
    scored = 0
    for r in rows:
        dcf_score = max(-SCORE_CAP, min(SCORE_CAP, r["dcf_score"]))
        relative_score = r.get("relative_score")
        if relative_score is None:
            composite = dcf_score
        else:
            relative_score = max(-SCORE_CAP, min(SCORE_CAP, relative_score))
            composite = dcf_weight * dcf_score + relative_weight * relative_score
        rating = rating_from_score(composite, band_width)
        result = score_rating(rating, r["realized_return_pct"])
        if result is None:
            continue
        scored += 1
        if result:
            correct += 1
    if scored == 0:
        return None
    return correct, scored, 100 * correct / scored


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_PATH
    rows = load_rows(path)
    print(f"Loaded {len(rows)} usable (DCF-available) rows from {path}", file=sys.stderr)

    results = []
    for band_width in BAND_WIDTHS:
        for dcf_weight in DCF_WEIGHTS:
            outcome = evaluate_config(rows, band_width, dcf_weight)
            if outcome is None:
                continue
            correct, scored, acc = outcome
            results.append({
                "band_width": band_width,
                "dcf_weight": dcf_weight,
                "relative_weight": round(1.0 - dcf_weight, 2),
                "correct": correct,
                "scored": scored,
                "accuracy": acc,
            })

    results.sort(key=lambda r: -r["accuracy"])

    print(f"\n{'BandWidth':>10}{'DCFWt':>8}{'RelWt':>8}{'Correct':>9}{'Scored':>8}{'Accuracy':>10}")
    print("-" * 53)
    for r in results:
        print(f"{r['band_width']:>10.1f}{r['dcf_weight']:>8.1f}{r['relative_weight']:>8.1f}"
              f"{r['correct']:>9}{r['scored']:>8}{r['accuracy']:>9.1f}%")

    print(f"\nCurrent production config: band_width=15.0, dcf_weight=0.6, relative_weight=0.4")
    current = next((r for r in results if r["band_width"] == 15.0 and r["dcf_weight"] == 0.6), None)
    if current:
        print(f"  -> scores {current['accuracy']:.1f}% ({current['correct']}/{current['scored']}) on this grid")

    best = results[0]
    print(f"\nBest in grid: band_width={best['band_width']}, dcf_weight={best['dcf_weight']}, "
          f"relative_weight={best['relative_weight']} -> {best['accuracy']:.1f}% "
          f"({best['correct']}/{best['scored']})")

    # Plateau check: how many DISTINCT configs land within 2 points of
    # the best accuracy? A robust signal has many neighbors close to
    # the winner; an overfit spike has few.
    near_best = [r for r in results if best["accuracy"] - r["accuracy"] <= 2.0]
    print(f"\n{len(near_best)} of {len(results)} tested configs are within 2 points of the best "
          f"({'looks like a robust plateau' if len(near_best) >= 10 else 'looks like a narrow spike -- treat with caution'}).")


if __name__ == "__main__":
    main()
