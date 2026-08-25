"""
combine_ml_training_sets.py

Concatenates every per-window ml_training_set_{universe}_asof{N}mo.csv
(scripts/build_ml_training_set.py, one file per (universe, as-of date)
run) into the single ml_training_set.csv
app/valuation/ml_valuation_classifier.py actually trains against.

Kept as a separate, dumb concatenation step rather than folded into
build_ml_training_set.py itself so each window's build can run
independently (parallel background jobs, partial re-runs after a
failure) without the combine step racing a still-in-progress sibling.

De-duplicates on (ticker, as_of_date) -- the only way a real duplicate
row can arise is re-running the exact same (universe, months-ago) pair
twice, since every other combination produces a distinct as_of_date.
Keeps the last-seen occurrence (a re-run supersedes the file it
overwrote).

Run: python scripts/combine_ml_training_sets.py
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "ml_training_set.csv"


def main():
    window_files = sorted(SCRIPT_DIR.glob("ml_training_set_*.csv"))
    if not window_files:
        print("No ml_training_set_*.csv window files found -- nothing to combine.", file=sys.stderr)
        return

    frames = []
    for path in window_files:
        # A window that got fully rate-limited (every ticker errored)
        # writes a genuinely empty file (0 bytes -- build_ml_training_set.py's
        # rows=[] produces a column-less DataFrame, which to_csv writes as
        # nothing at all) -- pandas.read_csv raises EmptyDataError on that,
        # not just returning an empty frame, so it's caught explicitly
        # rather than crashing the whole combine over one bad window.
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"  {path.name}: empty file (fully rate-limited run), skipped", file=sys.stderr)
            continue
        if df.empty:
            print(f"  {path.name}: 0 rows, skipped", file=sys.stderr)
            continue
        frames.append(df)
        print(f"  {path.name}: {len(df)} rows", file=sys.stderr)

    if not frames:
        print("Every window file was empty -- nothing to combine.", file=sys.stderr)
        return

    combined = pd.concat(frames, ignore_index=True)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["ticker", "as_of_date"], keep="last")

    combined.to_csv(OUTPUT_PATH, index=False)

    print(file=sys.stderr)
    print(f"Windows combined: {len(frames)}   Rows before dedup: {before_dedup}   "
          f"Rows after dedup: {len(combined)}", file=sys.stderr)
    print(f"Label distribution:\n{combined['realized_label'].value_counts()}", file=sys.stderr)
    print(f"\nSaved -> {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
