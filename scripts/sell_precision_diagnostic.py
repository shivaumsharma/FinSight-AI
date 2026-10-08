"""
sell_precision_diagnostic.py

Step 1 of the plan to fix FinSight's diagnosed Sell-call weakness
(EVALUATION.md's "honest finding": Buy precision 52.8-65% across every
backtest window, Sell as low as 15.6% -- yet the composite score treats
Buy/Hold/Sell symmetrically, BUY_THRESHOLD=+7.5 and SELL_THRESHOLD=-7.5
in report_data_builder.py, despite that asymmetric reliability).

Pure post-hoc analysis of the two already-saved broad-universe backtest
artifacts (n=972+935=1,907 -- the exact dataset behind EVALUATION.md's
canonical 38.6% number). No new network calls, no new valuations, no
model changes -- every field used here (composite_score, dcf_score,
relative_score, signal_disagreement, category, realized_return_pct) is
already stored per row from the original backtest run.

Three questions, cheapest/most information-dense first:

1. Does Sell precision vary by sector? A proxy for hypergrowth
   concentration, since the broad-universe file doesn't tag
   hypergrowth/negative-FCF status directly the way
   phase2_backtest.py's curated TICKERS dict does for the smaller
   curated set.
2. Does requiring DCF and relative valuation to AGREE before calling
   Sell (not just the 80/20 blend clearing SELL_THRESHOLD) raise
   precision -- using the `signal_disagreement` field already computed
   per row at backtest time.
3. Would a stricter (more negative) SELL_THRESHOLD raise Sell
   precision, or does raw accuracy only rise because Sell volume drops
   -- the exact trap scripts/tune_momentum_weight.py already caught
   once (accuracy up, precision on the affected class flat, volume
   down = base-rate artifact, not a real effect). Checked the same way
   that script checked it: against a same-size baseline, not accuracy
   alone -- here, many random same-size draws from the current
   (SELL_THRESHOLD=-7.5) Sell-eligible pool, since composite_score is
   the very axis being tightened.

Run: python scripts/sell_precision_diagnostic.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.baseline_scoring import score_rating

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_FILES = [
    "backtest_results_ticker_universe_asof12mo_exit0mo.json",
    "backtest_results_ticker_universe_asof24mo_exit12mo.json",
]

# Current production value is -7.5 (report_data_builder.SELL_THRESHOLD).
# Swept more negative only -- a *stricter* cutoff, matching the
# "don't sell as often, but be more right when you do" hypothesis.
SELL_THRESHOLDS = [-7.5, -10.0, -12.5, -15.0, -17.5, -20.0, -25.0, -30.0]
RANDOM_DRAWS = 2000
SEED = 42


def load_rows():
    rows = []
    for filename in RESULT_FILES:
        path = SCRIPT_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename}: not found", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        before = len(rows)
        rows.extend(
            r for r in data
            if not r.get("error") and r.get("realized_return_pct") is not None
        )
        print(f"  {filename}: {len(rows) - before} usable rows", file=sys.stderr)
    return rows


def precision_of(rows, as_rating):
    """Precision if every row in `rows` were scored as `as_rating`
    ("Buy"/"Sell") against its own realized_return_pct -- independent
    of what recommendation the row was actually stored with, so this
    doubles as both "precision of the live Sell calls" (pass the rows
    already recommendation=='Sell') and "precision of a retroactive,
    counterfactual Sell rule" (pass any other row subset)."""
    scored = [r for r in rows if r["realized_return_pct"] is not None]
    if not scored:
        return None, 0
    correct = sum(1 for r in scored if score_rating(as_rating, r["realized_return_pct"]))
    return 100 * correct / len(scored), len(scored)


def sanity_check(rows):
    print("\n--- Sanity check against EVALUATION.md's stated pooled figures ---", file=sys.stderr)
    for rating in ("Buy", "Sell", "Hold"):
        live_calls = [r for r in rows if r.get("recommendation") == rating]
        prec, n = precision_of(live_calls, rating)
        prec_str = f"{prec:.1f}%" if prec is not None else "--"
        print(f"  {rating:<5} n={n:<6} precision={prec_str}", file=sys.stderr)


def by_sector(rows):
    print("\n--- Question 1: Sell precision by sector (proxy for hypergrowth concentration, n>=5 only) ---", file=sys.stderr)
    cats = {}
    for r in rows:
        cats.setdefault(r.get("category", "Unknown"), []).append(r)

    header = f"{'Sector':<34}{'Sell N':>8}{'Sell Prec':>11}{'Buy N':>8}{'Buy Prec':>10}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    ranked = []
    for cat in sorted(cats):
        cat_rows = cats[cat]
        sell_calls = [r for r in cat_rows if r.get("recommendation") == "Sell"]
        buy_calls = [r for r in cat_rows if r.get("recommendation") == "Buy"]
        sell_prec, sell_n = precision_of(sell_calls, "Sell")
        buy_prec, buy_n = precision_of(buy_calls, "Buy")
        if sell_n >= 5:
            ranked.append((cat, sell_prec, sell_n))
        sell_str = f"{sell_prec:.1f}%" if sell_prec is not None else "--"
        buy_str = f"{buy_prec:.1f}%" if buy_prec is not None else "--"
        print(f"{cat:<34}{sell_n:>8}{sell_str:>11}{buy_n:>8}{buy_str:>10}", file=sys.stderr)

    ranked.sort(key=lambda t: t[1])
    if ranked:
        print(f"\nWorst-Sell-precision sector (n>=5): {ranked[0][0]} at {ranked[0][1]:.1f}% (n={ranked[0][2]})", file=sys.stderr)
        print(f"Best-Sell-precision sector (n>=5):  {ranked[-1][0]} at {ranked[-1][1]:.1f}% (n={ranked[-1][2]})", file=sys.stderr)


def by_agreement(rows):
    print("\n--- Question 2: Sell precision when DCF & relative valuation agree vs. disagree ---", file=sys.stderr)
    sell_calls = [r for r in rows if r.get("recommendation") == "Sell"]
    agree = [r for r in sell_calls if not r.get("signal_disagreement")]
    disagree = [r for r in sell_calls if r.get("signal_disagreement")]
    for label, subset in [
        ("Agree    (relative valuation also confirms overvalued)", agree),
        ("Disagree (DCF alone drove the Sell call)", disagree),
    ]:
        prec, n = precision_of(subset, "Sell")
        prec_str = f"{prec:.1f}%" if prec is not None else "--"
        print(f"  {label:<58} n={n:<5} precision={prec_str}", file=sys.stderr)


def threshold_sweep(rows):
    print("\n--- Question 3: retroactive SELL_THRESHOLD sweep (composite_score already stored, no re-scoring) ---", file=sys.stderr)
    rng = random.Random(SEED)

    scoreable = [r for r in rows if r.get("composite_score") is not None]
    current_pool = [r for r in scoreable if r["composite_score"] <= SELL_THRESHOLDS[0]]
    print(f"Current production SELL_THRESHOLD={SELL_THRESHOLDS[0]}: pool of {len(current_pool)} candidates to draw random-matched baselines from.", file=sys.stderr)

    header = f"{'Threshold':>10}{'Sell N':>9}{'Sell Prec':>11}{'Random-matched Prec (avg of ' + str(RANDOM_DRAWS) + ')':>38}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)

    for threshold in SELL_THRESHOLDS:
        would_sell = [r for r in scoreable if r["composite_score"] <= threshold]
        prec, n = precision_of(would_sell, "Sell")
        prec_str = f"{prec:.1f}%" if prec is not None else "--"

        if n == 0 or n >= len(current_pool):
            random_str = "-- (n/a at full pool size)"
        else:
            random_precisions = []
            for _ in range(RANDOM_DRAWS):
                sample = rng.sample(current_pool, n)
                r_prec, _ = precision_of(sample, "Sell")
                if r_prec is not None:
                    random_precisions.append(r_prec)
            random_avg = sum(random_precisions) / len(random_precisions) if random_precisions else None
            random_str = f"{random_avg:.1f}%" if random_avg is not None else "--"

        print(f"{threshold:>10.1f}{n:>9}{prec_str:>11}{random_str:>38}", file=sys.stderr)

    print(
        "\nReading this table: if a stricter threshold's Sell Prec clearly beats its own "
        "Random-matched Prec, tightening the composite-score cutoff is finding genuinely "
        "worse Sells to cut, not just shrinking volume. If the two columns track closely, "
        "this is the same base-rate trap tune_momentum_weight.py already caught once -- "
        "don't ship a threshold change on raw-precision-improved-at-a-tighter-cutoff alone.",
        file=sys.stderr,
    )


def sector_exclusion_check(rows):
    """Follow-up on Question 1's raw finding: Financials (worst Sell
    precision across all three size classes) and Energy stand out even
    by eye, but a raw subgroup average can't tell "real, structural
    effect" apart from "n happened to land favorably" -- same
    matched-size-random-baseline discipline as the threshold sweep."""
    print("\n--- Question 4: does excluding Financials/Energy Sells raise precision beyond random attrition? ---", file=sys.stderr)
    rng = random.Random(SEED)
    sell_calls = [r for r in rows if r.get("recommendation") == "Sell"]
    all_prec, all_n = precision_of(sell_calls, "Sell")
    print(f"  All Sells: n={all_n} precision={all_prec:.1f}%", file=sys.stderr)

    exclusions = {
        "Excluding Financials": lambda r: not r["category"].startswith("Financials"),
        "Excluding Financials + Energy": lambda r: not r["category"].startswith("Financials") and not r["category"].startswith("Energy"),
    }
    for label, keep in exclusions.items():
        subset = [r for r in sell_calls if keep(r)]
        prec, n = precision_of(subset, "Sell")
        random_precisions = []
        for _ in range(RANDOM_DRAWS):
            sample = rng.sample(sell_calls, n)
            r_prec, _ = precision_of(sample, "Sell")
            if r_prec is not None:
                random_precisions.append(r_prec)
        random_avg = sum(random_precisions) / len(random_precisions)
        gap = prec - random_avg
        verdict = "REAL SIGNAL" if gap >= 2.0 else "within noise of random"
        print(f"  {label:<32} n={n:<5} precision={prec:.1f}%   random-matched={random_avg:.1f}%   gap={gap:+.1f}pt -> {verdict}", file=sys.stderr)


def main():
    print("Loading broad-universe backtest artifacts...", file=sys.stderr)
    rows = load_rows()
    print(f"\nTotal usable rows: {len(rows)}", file=sys.stderr)

    sanity_check(rows)
    by_sector(rows)
    by_agreement(rows)
    threshold_sweep(rows)
    sector_exclusion_check(rows)


if __name__ == "__main__":
    main()
