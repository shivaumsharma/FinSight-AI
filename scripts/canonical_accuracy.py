"""
canonical_accuracy.py

FinSight's ONE reported accuracy number -- not a table of sub-metrics,
not a consensus-agreement score, one figure with a baseline and a
confidence interval.

Definition: of every Buy/Hold/Sell call FinSight's real production
decision path (app.reporting.report_data_builder.derive_recommendation
-- the exact function the deployed app calls, not a reimplementation)
made on the broad, non-cherry-picked S&P 500+400+600(partial) universe,
what fraction were correct 12 months later? "Correct" is the same rule
already tuned and validated in EVALUATION.md and unchanged here (see
app/analysis/baseline_scoring.py): Buy needs realized return > +5%,
Sell needs < -5%, Hold needs to land between the two.

Reported against the Always-Buy baseline on the identical cohort, with
sample size and a Wilson 95% CI -- a number with no baseline and no CI
is not a claim.

Why two already-computed backtest files, pooled, and not the live
call_tracker DB: this app deploys to free-tier hosts with an ephemeral
filesystem (see README's Cloud Run/Railway notes) -- jobs.db resets on
every redeploy, so a metric that depends on production calls surviving
across months of uptime is not reproducible on this project's actual
deploy target today. Both source files below are git-committed
artifacts that ship with the code and survive any redeploy:
  - backtest_results_ticker_universe_asof12mo_exit0mo.json (call made
    12mo ago, checked today -- a 12-month horizon)
  - backtest_results_ticker_universe_asof24mo_exit12mo.json (call made
    24mo ago, checked 12mo ago -- also a 12-month horizon, just an
    earlier vintage)
Both are genuine 12-month-horizon evaluations of the SAME rule against
the SAME broad universe, so pooling them is combining two honest
samples of one metric, not averaging apples and oranges.

"Scaling this up" going forward means re-running phase2_backtest.py
against new/rolling historical windows (a bear-market window, a later
vintage, more tickers) and re-running this script -- growing the
sample and the diversity of regimes tested through more backtest runs,
not through waiting on production traffic an ephemeral host can't
retain. If a persistent volume is ever attached (README already
documents how, for Cloud Run/Railway), the live call_tracker's
window_days=365 checkpoints (app/api/db.py) can be pooled in here too
via the same score_rating/naive_baseline_accuracy functions -- the
plumbing for that already exists, it's just not load-bearing today.

Output: prints a human-readable report and writes
scripts/canonical_accuracy_result.json, the one file
report_data_builder.py reads to surface this number on every generated
report (see build_report_data's "track_record" field).

Run: python scripts/canonical_accuracy.py
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.baseline_scoring import score_rating

SCRIPT_DIR = Path(__file__).resolve().parent

SOURCE_FILES = [
    "backtest_results_ticker_universe_asof12mo_exit0mo.json",
    "backtest_results_ticker_universe_asof24mo_exit12mo.json",
]

OUTPUT_PATH = SCRIPT_DIR / "canonical_accuracy_result.json"

Z_95 = 1.959963984540054


def wilson_interval(correct: int, n: int, z: float = Z_95):
    """95% Wilson score interval for a binomial proportion -- safe at
    small N and never produces an out-of-[0,1] bound the way a naive
    normal-approximation interval can."""
    if n == 0:
        return None, None
    phat = correct / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    spread = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    low = (center - spread) / denom
    high = (center + spread) / denom
    return max(0.0, low), min(1.0, high)


def load_source(filename: str):
    path = SCRIPT_DIR / filename
    with open(path) as f:
        rows = json.load(f)
    # Only rows with a real rating and a real realized return are
    # scoreable -- errored/"Insufficient Data" rows contribute nothing
    # to either the model's score or the baseline's, same exclusion
    # score_rating() itself already applies.
    scoreable = [
        r for r in rows
        if r.get("recommendation") not in (None, "Insufficient Data")
        and r.get("realized_return_pct") is not None
    ]
    return scoreable


def main():
    pooled_rows = []
    sources_meta = []

    for filename in SOURCE_FILES:
        rows = load_source(filename)
        pooled_rows.extend(rows)
        as_of_dates = sorted({r["as_of_date"] for r in rows if r.get("as_of_date")})
        sources_meta.append({
            "file": filename,
            "n_scored": len(rows),
            "as_of_range": [as_of_dates[0], as_of_dates[-1]] if as_of_dates else None,
        })

    n = len(pooled_rows)
    model_correct = sum(
        1 for r in pooled_rows
        if score_rating(r["recommendation"], r["realized_return_pct"])
    )
    buy_correct = sum(
        1 for r in pooled_rows
        if score_rating("Buy", r["realized_return_pct"])
    )

    model_pct = 100 * model_correct / n
    buy_pct = 100 * buy_correct / n
    model_lo, model_hi = wilson_interval(model_correct, n)
    buy_lo, buy_hi = wilson_interval(buy_correct, n)

    result = {
        "metric": "12-month forward directional accuracy",
        "methodology": (
            "Buy/Hold/Sell calls from FinSight's real production decision "
            "path (report_data_builder.derive_recommendation), scored "
            "12 months later against the same +-5%% rule validated in "
            "EVALUATION.md. Pooled across two non-overlapping historical "
            "windows on the broad 1,002-ticker S&P 500+400+600(partial) "
            "universe -- not a hand-picked ticker list."
        ),
        "n": n,
        "model_accuracy_pct": round(model_pct, 1),
        "model_ci_95": [round(model_lo * 100, 1), round(model_hi * 100, 1)],
        "always_buy_baseline_pct": round(buy_pct, 1),
        "always_buy_ci_95": [round(buy_lo * 100, 1), round(buy_hi * 100, 1)],
        "beats_baseline": model_pct > buy_pct,
        "sources": sources_meta,
        "live_tracker_n": 0,
        "live_tracker_note": (
            "The live production call tracker (app/api/db.py "
            "tracked_call_checkpoints, window_days=365) is not pooled in "
            "yet -- this deploy target's filesystem is ephemeral across "
            "redeploys, so it has no data that has reliably survived to "
            "maturity. See this script's module docstring."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproduce": "python scripts/canonical_accuracy.py",
    }
    result["summary_line"] = (
        f"{result['model_accuracy_pct']}% forward-accuracy "
        f"(N={n}, 95% CI {result['model_ci_95'][0]}-{result['model_ci_95'][1]}%) "
        f"vs. {result['always_buy_baseline_pct']}% Always-Buy baseline "
        f"(95% CI {result['always_buy_ci_95'][0]}-{result['always_buy_ci_95'][1]}%)"
    )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"FinSight canonical accuracy metric -- {result['metric']}\n")
    print(f"  {result['summary_line']}")
    print(f"  Model {'beats' if result['beats_baseline'] else 'loses to'} the naive baseline.\n")
    for s in sources_meta:
        print(f"  - {s['file']}: n={s['n_scored']}, as-of range {s['as_of_range']}")
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
