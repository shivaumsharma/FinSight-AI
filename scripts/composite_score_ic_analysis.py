"""
composite_score_ic_analysis.py

Root-causes a specific question the walk-forward backtest's own result
raised (EVALUATION.md section 9): the composite-score-driven top-25
strategy underperformed a no-signal naive baseline on EVERY metric. Is
that because composite_score carries weak-to-no genuine forward-
predictive power at all (a signal problem -- the ranking itself is
noise or inverted), or because concentrating into only 25 names
amplified idiosyncratic risk on top of a ranking that's directionally
fine on average (a diversification/sizing problem)? Different
diagnoses need different fixes, and guessing which one it is instead
of measuring would repeat the exact mistake this project's own
evaluation culture (EVALUATION.md throughout) exists to avoid.

Method: a standard quant "Information Coefficient" analysis. Reuses
the exact same universe sample, rebalance dates, and point-in-time
scoring as walkforward_backtest.py (same functions, same seed) -- but
instead of only tracking the top-25 subset that actually got traded,
scores and tracks the FULL sampled universe at every rebalance date,
pairs each ticker's composite_score (and its DCF-only/relative-only
component scores) with its REALIZED return over the following holding
period, and computes:
  - Pooled Spearman rank IC (composite_score vs. forward return)
    across every (ticker, period) pair, and per-period IC (mean/std
    across the 12 periods -- the more standard quant-research framing,
    since a single pooled IC can be dominated by whichever period had
    the most dispersion).
  - The same IC decomposed for dcf_score and relative_score
    separately -- isolates whether either leg alone carries real
    signal, and whether DCF_WEIGHT=0.8/RELATIVE_WEIGHT=0.2
    (report_data_builder.py) is over-weighting the weaker of the two.
  - A decile table: pooled mean AND stdev of forward return by
    composite_score decile -- a working ranking signal shows a
    monotonically increasing mean across deciles; a broken one shows a
    flat or inverted pattern. The stdev column speaks directly to the
    "was concentration amplifying noise" half of the question above.

Run: python scripts/composite_score_ic_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from scripts.phase2_backtest import MARKET_BENCHMARK, _price_on_or_before, _score_ticker_at_date, _tz_naive
from scripts.walkforward_backtest import UNIVERSE_SAMPLE_SEED, _fetch_all, _rebalance_dates, _sample_universe

SCRIPT_DIR = Path(__file__).resolve().parent
UNIVERSE_SIZE = 275
N_DECILES = 10


def _build_period_rows(rebalance_dates, raw_by_ticker, categories, market_history, tnx_history):
    """One row per (ticker, holding period) with the score as of the
    period's start date and the realized return over that period --
    the full universe, not just whatever a downstream strategy would
    have selected."""
    rows = []
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]
        print(f"  scoring period {i+1}/{len(rebalance_dates)-1} ({start_date.date()} -> {end_date.date()})...",
              file=sys.stderr)
        for ticker, raw_data in raw_by_ticker.items():
            try:
                row = _score_ticker_at_date(
                    ticker, categories[ticker], raw_data, start_date, start_date, market_history, tnx_history
                )
            except Exception:
                continue
            if row.get("composite_score") is None:
                continue

            price_history = raw_data["price_history"]
            price_start = _price_on_or_before(price_history, start_date)
            price_end = _price_on_or_before(price_history, end_date)
            if not price_start or not price_end:
                continue
            forward_return_pct = (price_end - price_start) / price_start * 100

            rows.append({
                "ticker": ticker,
                "category": categories[ticker],
                "period_index": i,
                "as_of_date": start_date.date().isoformat(),
                "composite_score": row["composite_score"],
                "dcf_score": row.get("dcf_score"),
                "relative_score": row.get("relative_score"),
                "recommendation": row.get("recommendation"),
                "forward_return_pct": forward_return_pct,
            })
    return pd.DataFrame(rows)


def _pooled_and_per_period_ic(df: pd.DataFrame, score_col: str) -> dict:
    sub = df.dropna(subset=[score_col, "forward_return_pct"])
    pooled_ic = sub[score_col].corr(sub["forward_return_pct"], method="spearman")

    per_period = []
    for period_index, group in sub.groupby("period_index"):
        if len(group) < 10:
            continue
        ic = group[score_col].corr(group["forward_return_pct"], method="spearman")
        if pd.notna(ic):
            per_period.append(ic)

    return {
        "pooled_ic": float(pooled_ic) if pd.notna(pooled_ic) else None,
        "mean_period_ic": float(np.mean(per_period)) if per_period else None,
        "std_period_ic": float(np.std(per_period)) if per_period else None,
        "n_periods_with_ic": len(per_period),
        "n_rows": len(sub),
    }


def _decile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns deciles WITHIN each rebalance period (not pooled first)
    -- composite_score isn't on a stable absolute scale across periods
    (it's a percentile-rank score against a fixed historical reference
    table, report_data_builder.py's own comment, but the CROSS-
    SECTIONAL distribution within one date is what a ranking strategy
    actually trades on), then pools the resulting decile buckets
    across all periods for the final mean/stdev."""
    def _assign_deciles(group):
        if len(group) < N_DECILES:
            return pd.Series([None] * len(group), index=group.index)
        return pd.qcut(group["composite_score"], N_DECILES, labels=False, duplicates="drop")

    df = df.copy()
    df["decile"] = df.groupby("period_index", group_keys=False).apply(_assign_deciles, include_groups=False)
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int) + 1  # 1 = lowest score, 10 = highest

    table = df.groupby("decile")["forward_return_pct"].agg(["mean", "std", "count"]).reset_index()
    return table


def main():
    today_date = pd.Timestamp(datetime.now().date())
    rebalance_dates = _rebalance_dates(today_date)
    categories = _sample_universe(UNIVERSE_SIZE, UNIVERSE_SAMPLE_SEED)

    print(f"Universe: {len(categories)} tickers, {len(rebalance_dates)} rebalance dates "
          f"({rebalance_dates[0].date()} -> {rebalance_dates[-1].date()})", file=sys.stderr)

    print("Fetching market/risk-free-rate history...", file=sys.stderr)
    market_history = _tz_naive(yf.Ticker(MARKET_BENCHMARK).history(period="6y"))
    tnx_history = _tz_naive(yf.Ticker("^TNX").history(period="6y"))

    print(f"Fetching {len(categories)} tickers...", file=sys.stderr)
    raw_by_ticker = _fetch_all(categories, workers=10)
    print(f"{len(raw_by_ticker)}/{len(categories)} fetched successfully", file=sys.stderr)

    df = _build_period_rows(rebalance_dates, raw_by_ticker, categories, market_history, tnx_history)
    print(f"\nTotal (ticker, period) rows scored: {len(df)}", file=sys.stderr)

    df.to_csv(SCRIPT_DIR / "composite_score_ic_rows.csv", index=False)

    print("\n=== Information Coefficient (Spearman rank correlation vs. forward return) ===", file=sys.stderr)
    for label, col in [("composite_score", "composite_score"), ("dcf_score", "dcf_score"), ("relative_score", "relative_score")]:
        result = _pooled_and_per_period_ic(df, col)
        pooled = f"{result['pooled_ic']:+.3f}" if result["pooled_ic"] is not None else "--"
        mean_p = f"{result['mean_period_ic']:+.3f}" if result["mean_period_ic"] is not None else "--"
        std_p = f"{result['std_period_ic']:.3f}" if result["std_period_ic"] is not None else "--"
        print(f"  {label:<16} pooled IC={pooled}   mean-per-period IC={mean_p} (std {std_p}, "
              f"n={result['n_periods_with_ic']} periods, {result['n_rows']} rows)", file=sys.stderr)

    print("\n=== Decile table: forward return by composite_score decile (1=lowest score, 10=highest) ===", file=sys.stderr)
    decile_table = _decile_table(df)
    print(f"{'Decile':>7}{'Mean fwd return':>18}{'Stdev':>10}{'N':>8}", file=sys.stderr)
    for _, r in decile_table.iterrows():
        print(f"{int(r['decile']):>7}{r['mean']:>17.2f}%{r['std']:>9.2f}%{int(r['count']):>8}", file=sys.stderr)

    decile_table.to_csv(SCRIPT_DIR / "composite_score_decile_table.csv", index=False)
    print(f"\nSaved -> {SCRIPT_DIR / 'composite_score_ic_rows.csv'}, {SCRIPT_DIR / 'composite_score_decile_table.csv'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
