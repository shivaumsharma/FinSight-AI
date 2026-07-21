"""
validate_tuning_stability.py

Answers the question tune_recommendation_config.py's grid search left
open: is band_width=5.0/dcf_weight=1.0 (the top result on the full
40-row DCF-available sample) a real pattern, or noise from a small
sample?

Method: repeated random subsampling validation. Many times over, split
the 40 rows into a random train subset and a held-out test subset,
find the best (band_width, dcf_weight) using ONLY the train subset
(exactly like tune_recommendation_config.py's grid search, just on
less data), then score that chosen config on the test subset it never
saw. Compare that against how the current production config
(band=15, dcf_weight=0.6) scores on the same held-out test subsets --
a fair head-to-head on data neither config was picked using.

If the same config keeps winning on train AND keeps scoring well on
held-out test across trials, that's real signal. If a different
config wins nearly every trial, or the "winner" scores no better than
production on held-out data, that confirms the single-grid-search
result was overfit to this specific 40-row sample.

No network calls -- reuses scripts/backtest_results.json, same as
tune_recommendation_config.py.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tune_recommendation_config import (
    RESULTS_PATH, BAND_WIDTHS, DCF_WEIGHTS, SCORE_CAP,
    score_rating, rating_from_score, load_rows,
)

N_TRIALS = 500
TRAIN_FRACTION = 0.7
RANDOM_SEED = 42

# Current production config, for the head-to-head comparison -- the
# values actually shipped in report_data_builder.py right now (band=
# 7.5, dcf_weight=0.8), not the pre-this-session baseline (15, 0.6).
PROD_BAND_WIDTH = 7.5
PROD_DCF_WEIGHT = 0.8


def score_subset(rows, band_width, dcf_weight):
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
    return correct / scored, scored


def best_config_on(rows):
    best = None
    for band_width in BAND_WIDTHS:
        for dcf_weight in DCF_WEIGHTS:
            outcome = score_subset(rows, band_width, dcf_weight)
            if outcome is None:
                continue
            acc, scored = outcome
            if best is None or acc > best[2]:
                best = (band_width, dcf_weight, acc, scored)
    return best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    rows = load_rows(path)
    n = len(rows)
    train_size = round(n * TRAIN_FRACTION)
    print(f"Loaded {n} rows. {N_TRIALS} trials, {train_size} train / {n - train_size} test per trial.",
          file=sys.stderr)

    rng = random.Random(RANDOM_SEED)

    winner_counts = Counter()
    winner_test_accs = []
    prod_test_accs = []

    for _ in range(N_TRIALS):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        train_rows = shuffled[:train_size]
        test_rows = shuffled[train_size:]

        best = best_config_on(train_rows)
        if best is None:
            continue
        band_width, dcf_weight, train_acc, _ = best
        winner_counts[(band_width, dcf_weight)] += 1

        test_outcome = score_subset(test_rows, band_width, dcf_weight)
        if test_outcome is not None:
            winner_test_accs.append(test_outcome[0])

        prod_outcome = score_subset(test_rows, PROD_BAND_WIDTH, PROD_DCF_WEIGHT)
        if prod_outcome is not None:
            prod_test_accs.append(prod_outcome[0])

    print(f"\nHow often each config won on the training split (top 10 of {len(winner_counts)} distinct winners):")
    print(f"{'BandWidth':>10}{'DCFWt':>8}{'Win %':>9}")
    print("-" * 27)
    for (band_width, dcf_weight), count in winner_counts.most_common(10):
        print(f"{band_width:>10.1f}{dcf_weight:>8.1f}{100*count/N_TRIALS:>8.1f}%")

    def avg(values):
        return 100 * sum(values) / len(values) if values else float("nan")

    print(f"\nHeld-out (test-split) accuracy, averaged across {len(winner_test_accs)} trials:")
    print(f"  Whatever config won on that trial's train split:  {avg(winner_test_accs):.1f}%")
    print(f"  Current production config (band={PROD_BAND_WIDTH}, dcf_weight={PROD_DCF_WEIGHT}), "
          f"same test splits:  {avg(prod_test_accs):.1f}%")

    diff = avg(winner_test_accs) - avg(prod_test_accs)
    print(f"\nDifference: {diff:+.1f} points")
    if abs(diff) < 3.0:
        print("-> Essentially no difference on unseen data. The single-grid-search winner does NOT "
              "generalize -- confirms it was overfit to the full 40-row sample, not a real pattern.")
    elif diff > 0:
        print("-> The winning config keeps beating production on data it wasn't chosen from -- "
              "real signal, not just noise (though still only one 12-month window).")
    else:
        print("-> The winning config actually does WORSE than production on unseen data -- "
              "the single-grid-search result was overfit.")


if __name__ == "__main__":
    main()
