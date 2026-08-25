"""
cross_sectional_ranking.py

Phase 6: does composite_score have real cross-sectional ranking skill
-- does a higher score correspond to a higher realized return, across
the whole universe at a point in time -- independent of the fixed
+/-7.5 Buy/Hold/Sell threshold?

Motivation: EVALUATION.md's own "return spread" metric (Buy avg return
minus Sell avg return) flips sign across windows -- positive in one,
negative in four others, including the least-bullish window tested.
That metric only uses the tickers that crossed the Buy/Sell threshold,
discarding every Hold call and collapsing a continuous score into 3
buckets. Spearman rank correlation and decile spread use the full
distribution instead -- a genuinely different (and more standard, in
the quant-factor literature) question: if you ranked the whole
universe by composite_score, would the top of that ranking actually
outperform the bottom?

Uses backtest_results_*.json files ALREADY on disk (composite_score
and realized_return_pct are both saved per row already) -- no new
network calls, no new backtest run. Every window computed and
committed so far this session is included.

Run: python scripts/cross_sectional_ranking.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent

WINDOW_FILES = [
    "backtest_results_curated_asof12mo_exit0mo.json",
    "backtest_results_curated_asof24mo_exit12mo.json",
    "backtest_results_curated_asof20mo_exit8mo.json",
    "backtest_results_ticker_universe_asof12mo_exit0mo.json",
    "backtest_results_ticker_universe_asof24mo_exit12mo.json",
]

N_BUCKETS = 5  # quintiles -- deciles would need more usable rows per window than the smaller curated sets have


def load_scoreable_rows(filename: str):
    path = SCRIPT_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        rows = json.load(f)
    return [
        r for r in rows
        if r.get("composite_score") is not None and r.get("realized_return_pct") is not None
    ]


def quintile_spread(rows):
    """Sorts by composite_score ascending, splits into N_BUCKETS
    roughly-equal groups, reports each group's mean realized_return_pct.
    A genuinely informative score should show a MONOTONIC increase from
    bucket 1 (lowest score) to bucket N (highest score) -- the classic
    quant-factor "does the top decile beat the bottom decile" check."""
    sorted_rows = sorted(rows, key=lambda r: r["composite_score"])
    buckets = np.array_split(sorted_rows, N_BUCKETS)
    means = []
    for bucket in buckets:
        if len(bucket) == 0:
            means.append(None)
            continue
        returns = [r["realized_return_pct"] for r in bucket]
        means.append(sum(returns) / len(returns))
    return means


def main():
    print(f"{'Window':<55}{'n':>6}{'Spearman rho':>14}{'p-value':>10}", file=sys.stderr)
    print("-" * 85, file=sys.stderr)

    all_rho = []
    for filename in WINDOW_FILES:
        rows = load_scoreable_rows(filename)
        if not rows:
            print(f"{filename:<55}{'--':>6}{'(no data)':>14}", file=sys.stderr)
            continue

        scores = [r["composite_score"] for r in rows]
        returns = [r["realized_return_pct"] for r in rows]
        rho, p_value = spearmanr(scores, returns)
        all_rho.append(rho)

        label = filename.replace("backtest_results_", "").replace(".json", "")
        print(f"{label:<55}{len(rows):>6}{rho:>+14.3f}{p_value:>10.4f}", file=sys.stderr)

        means = quintile_spread(rows)
        formatted = "  ".join(f"{m:+.1f}%" if m is not None else "n/a" for m in means)
        monotonic = all(
            a is None or b is None or a <= b
            for a, b in zip(means, means[1:])
        )
        print(f"    Quintile avg return (lowest score -> highest score): {formatted}"
              f"   [{'monotonic increasing' if monotonic else 'NOT monotonic'}]", file=sys.stderr)

    if all_rho:
        print(f"\nMean Spearman rho across {len(all_rho)} windows: {np.mean(all_rho):+.3f}"
              f"  (range {min(all_rho):+.3f} to {max(all_rho):+.3f})", file=sys.stderr)
        print(
            "Interpretation: rho near 0 means the composite score has no real cross-sectional "
            "ranking power -- a ticker scored higher is no more likely to outperform a ticker "
            "scored lower, regardless of whether either crosses the Buy/Sell threshold. A "
            "consistently positive rho across windows (not just one) would be the actual "
            "evidence of ranking skill the sign-flipping return-spread metric couldn't provide.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
