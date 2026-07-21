"""
rank_evaluation.py

Market-neutral evaluation of the DCF valuation signal -- a different
question than phase2_backtest.py's own accuracy score asks.

Why this exists: phase2_backtest.py scores each individual call
against a fixed +/-5% band ("was this specific Buy/Hold/Sell right").
That score stayed essentially flat (34.6% -> 35.0% -> 34.9%) across
three real changes to the DCF this session (see valuation_pipeline.py's
DEFAULT_TERMINAL_GROWTH_RATE comment and fcff_engine.py's
_quality_hold_years for what changed and why) -- and it's structurally
biased by testing over one 12-month window where the broad market rose
58.6% of the time regardless of any stock's fundamentals, so a
fixed-band score can't distinguish "the model has no skill" from "the
model has real skill but this window was an unusually one-directional
market."

What this asks instead: regardless of which direction the OVERALL
market moved, did the model's cheap-vs-expensive RANKING have real
information -- did stocks it called relatively cheap actually
outperform stocks it called relatively expensive, on average? Ranking
every company by the signal and comparing outcomes bucket-by-bucket
(the standard cross-sectional method in quant equity research) is
market-neutral by construction: it only asks whether the ORDERING was
right, not whether any individual call cleared an absolute return
threshold in a market that happened to be trending one way.

Signal: upside_pct (model intrinsic value vs. current price, i.e.
(intrinsic - price) / price * 100 -- POSITIVE means the model calls
the stock undervalued/cheap, exactly as computed live in
app/tools/valuation_tool.py). Outcome: realized_return_pct, the same
simple forward price return phase2_backtest.py already computed for
every name in a single run, from the SAME as_of_date to the SAME exit
date for every ticker in that run (verified below, not assumed) -- so
there is no per-name lookahead or inconsistent window to control for.

Uses already-fetched backtest data (scripts/backtest_results_*.json)
-- no network calls, no re-running the DCF pipeline, matching
tune_recommendation_config.py's own offline-recomputation approach.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent

# The three model states produced this session, in order. Each file is
# one phase2_backtest.py run (fixed as_of_date/exit_date across all
# ~1000 tickers in that run) against the same ticker_universe.json.
STATES = [
    ("Baseline", "Baseline (3% terminal growth, 4pt WACC floor spread, fixed 10yr fade)",
     SCRIPT_DIR / "backtest_results_ticker_universe_asof12mo_exit0mo_baseline.json"),
    ("Step 1", "Step 1 (4% terminal growth, 3pt WACC floor spread)",
     SCRIPT_DIR / "backtest_results_ticker_universe_asof12mo_exit0mo_step1.json"),
    ("Step 1+2", "Step 1+2 (+ ROE-based variable growth-fade duration)",
     SCRIPT_DIR / "backtest_results_ticker_universe_asof12mo_exit0mo.json"),
]

N_QUINTILES = 5
N_DECILES = 10


def load_signal_outcome_pairs(path):
    """
    Returns (pairs, total, dropped_no_signal, dropped_no_outcome,
    as_of_dates_seen). pairs is a list of (ticker, upside_pct,
    realized_return_pct) for every row with both a usable signal and a
    usable outcome. as_of_dates_seen is the set of distinct as_of_date
    values across all rows in the file -- should always be a single
    value (one backtest run = one as-of date for every ticker); this
    is checked explicitly by main() rather than assumed.
    """
    with open(path) as f:
        rows = json.load(f)

    total = len(rows)
    dropped_no_signal = 0
    dropped_no_outcome = 0
    pairs = []
    as_of_dates_seen = set()

    for r in rows:
        if r.get("as_of_date"):
            as_of_dates_seen.add(r["as_of_date"])

        if r.get("error"):
            dropped_no_signal += 1
            continue

        upside = r.get("upside_pct")
        outcome = r.get("realized_return_pct")

        if upside is None or upside != upside:  # NaN check (NaN != NaN)
            dropped_no_signal += 1
            continue
        if outcome is None or outcome != outcome:
            dropped_no_outcome += 1
            continue

        pairs.append((r["ticker"], upside, outcome))

    return pairs, total, dropped_no_signal, dropped_no_outcome, as_of_dates_seen


def bucket_analysis(pairs, n_buckets):
    """
    Sorts descending by signal (upside_pct) so bucket 1 = most
    undervalued (highest upside_pct = model says most upside to fair
    value) and bucket N = most overvalued (lowest/most-negative
    upside_pct). np.array_split gives the first (n % n_buckets)
    buckets one extra row when n doesn't divide evenly -- "roughly
    equal," not silently dropping remainder rows.
    """
    sorted_pairs = sorted(pairs, key=lambda p: -p[1])
    n = len(sorted_pairs)
    indices = np.array_split(np.arange(n), n_buckets)

    bucket_rows = []
    for i, idx in enumerate(indices):
        bucket_pairs = [sorted_pairs[j] for j in idx]
        returns = [p[2] for p in bucket_pairs]
        signal_vals = [p[1] for p in bucket_pairs]
        bucket_rows.append({
            "bucket": i + 1,
            "n": len(bucket_pairs),
            "signal_range": (min(signal_vals), max(signal_vals)),
            "mean_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
        })

    spread = bucket_rows[0]["mean_return"] - bucket_rows[-1]["mean_return"]
    return bucket_rows, spread


def information_coefficient(pairs):
    signals = [p[1] for p in pairs]
    outcomes = [p[2] for p in pairs]
    ic, p_value = spearmanr(signals, outcomes)
    return float(ic), float(p_value)


def print_bucket_table(label, bucket_rows, spread):
    print(f"\n{label}")
    print(f"{'Bucket':<8}{'N':>5}{'SignalRange%':>24}{'MeanReturn%':>14}{'MedianReturn%':>16}")
    print("-" * 67)
    for b in bucket_rows:
        lo, hi = b["signal_range"]
        print(f"{b['bucket']:<8}{b['n']:>5}{f'{lo:+.1f} to {hi:+.1f}':>24}"
              f"{b['mean_return']:>13.2f}%{b['median_return']:>15.2f}%")
    bottom = bucket_rows[-1]["bucket"]
    direction = "cheap beats expensive -- real signal" if spread > 0 else "expensive beats cheap -- inverted / no signal"
    print(f"Spread (bucket 1 minus bucket {bottom}): {spread:+.2f} points  ({direction})")


def analyze_file(short_label, label, path):
    """Runs the full quintile/decile/IC analysis on one backtest
    results file and prints it. Returns a summary dict for the
    cross-state comparison table, or None if the file was skipped."""
    print(f"\n{'=' * 90}")
    print(f"=== {label} ===")
    print(f"{'=' * 90}")

    if not path.exists():
        print(f"[skip] file not found: {path}", file=sys.stderr)
        return None

    pairs, total, dropped_signal, dropped_outcome, as_of_dates = load_signal_outcome_pairs(path)

    print(f"Total rows: {total}   Usable (signal+outcome): {len(pairs)}   "
          f"Dropped (no/NaN signal): {dropped_signal}   Dropped (no/NaN outcome): {dropped_outcome}")

    # Guardrail: every row in one backtest run should share the same
    # as_of_date -- confirms there's no accidental mixing of windows
    # within a single file, which would break the "same exit date for
    # every name" comparability this analysis relies on.
    if len(as_of_dates) == 1:
        print(f"As-of date check: OK, single as-of date across all rows ({next(iter(as_of_dates))})")
    else:
        print(f"As-of date check: WARNING -- {len(as_of_dates)} distinct as-of dates found: {as_of_dates}",
              file=sys.stderr)

    if len(pairs) < N_DECILES:
        print("[skip] too few usable rows for decile analysis", file=sys.stderr)
        return None

    quintile_rows, quintile_spread = bucket_analysis(pairs, N_QUINTILES)
    print_bucket_table("Quintiles (5 buckets)", quintile_rows, quintile_spread)

    decile_rows, decile_spread = bucket_analysis(pairs, N_DECILES)
    print_bucket_table("Deciles (10 buckets)", decile_rows, decile_spread)

    ic, p_value = information_coefficient(pairs)
    sig = "significant at 5%" if p_value < 0.05 else "NOT significant at 5%"
    print(f"\nInformation Coefficient (Spearman rank correlation, signal vs. forward return): "
          f"{ic:+.3f}  (p={p_value:.4f}, {sig}, n={len(pairs)})")

    return {
        "short_label": short_label, "n": len(pairs),
        "quintile_spread": quintile_spread, "decile_spread": decile_spread,
        "ic": ic, "p_value": p_value,
    }


def print_summary_table(summary_rows):
    print(f"\n{'=' * 90}")
    print("Cross-state comparison (headline numbers)")
    print(f"{'=' * 90}")
    print(f"{'State':<20}{'N':>6}{'QuintileSpread':>17}{'DecileSpread':>15}{'IC':>9}{'p-value':>10}")
    print("-" * 77)
    for s in summary_rows:
        print(f"{s['short_label']:<20}{s['n']:>6}{s['quintile_spread']:>16.2f}%{s['decile_spread']:>14.2f}%"
              f"{s['ic']:>+9.3f}{s['p_value']:>10.4f}")


def main():
    if len(sys.argv) > 1:
        # Ad-hoc single-file mode, e.g. to check a different backtest
        # window against the current model state: `rank_evaluation.py
        # path/to/backtest_results_X.json "My Label"`.
        path = Path(sys.argv[1])
        label = sys.argv[2] if len(sys.argv) > 2 else path.stem
        summary = analyze_file(label, label, path)
        if summary:
            print_summary_table([summary])
        return

    summary_rows = [
        row for row in (analyze_file(short_label, label, path) for short_label, label, path in STATES)
        if row is not None
    ]
    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
