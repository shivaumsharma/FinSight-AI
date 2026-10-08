"""
learned_portfolio.py -- RESEARCH SANDBOX. P8 + P9 + P11 for the learned 4-quarter model (A1b), DESCRIPTIVE.

Does a portfolio built from the model's walk-forward predictions beat holding every stock equally, after costs, and does it
survive a correction for the number of things tried?
  - Each quarter-end from 2016-06 the model ranks stocks (predictions made only with data known then).
  - Strategy: hold the top decile (by prediction), equal weight, for ONE quarter, then re-rank. Benchmark: equal-weight all
    stocks with a prediction that date. Delisted names keep their terminal return (see delisting.py).
  - Costs: one-way cost in basis points on the traded fraction (turnover), swept over 10 / 25 / 50 bps.
  - Deflated Sharpe ratio (Bailey & Lopez de Prado): probability the true Sharpe exceeds what the best of N_TRIALS random
    strategies would show by luck. Uses the full sprint trial count; the cross-trial Sharpe spread is approximated by the
    standard error of a Sharpe estimate (an assumption, stated).

Run: python research/learned_portfolio.py       (needs research/data/a1b_predictions.parquet from ml_cross_section_4q.py)
"""
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "data"
N_TRIALS = 33          # every variant counted in SPRINT_TRACKER.md at the time of this analysis
EULER = 0.5772156649
PERIODS_PER_YEAR = 4


def sharpe(x):
    return x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan


def deflated_sharpe(returns, n_trials):
    """P(true Sharpe > expected maximum Sharpe of n_trials luck-only strategies). Per-period (quarterly) units."""
    r = pd.Series(returns).dropna()
    T, sr = len(r), sharpe(r)
    sigma = sqrt((1 + 0.5 * sr ** 2) / (T - 1))                      # standard error of a Sharpe estimate (assumed spread across trials)
    sr0 = sigma * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * np.e)))
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * sqrt(T - 1) / denom)), sr0


def max_drawdown(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main():
    d = pd.read_parquet(DATA / "a1b_predictions.parquet")
    dates = sorted(d.date.unique())
    for model in ("pred_hgb", "pred_ridge"):
        held_prev, rows = set(), []
        for t in dates:
            g = d[(d.date == t) & d[model].notna() & d.fwd.notna()]
            if len(g) < 100:
                continue
            top = g.nlargest(max(10, len(g) // 10), model)
            held = set(top.ticker)
            turnover = 1.0 if not held_prev else 1 - len(held & held_prev) / len(held)   # fraction of the book replaced
            rows.append({"date": t, "gross": top.fwd.mean(), "bench": g.fwd.mean(), "turnover": turnover, "n_top": len(top)})
            held_prev = held
        r = pd.DataFrame(rows).set_index("date")
        print(f"\n=== {model}: top-decile long-only, quarterly rebalance, {len(r)} quarters {r.index.min().date()} -> {r.index.max().date()} ===")
        print(f"average turnover per rebalance {r.turnover.mean():.0%}; benchmark = equal-weight all scored stocks")
        out = []
        for bps in (0, 10, 25, 50):
            net = r.gross - r.turnover * 2 * bps / 1e4                       # round-trip on the replaced fraction
            ex = net - r.bench
            dsr, sr0 = deflated_sharpe(net, N_TRIALS)
            out.append({"cost_bps": bps, "ann_return%": (1 + net).prod() ** (PERIODS_PER_YEAR / len(net)) * 100 - 100,
                        "bench_ann%": (1 + r.bench).prod() ** (PERIODS_PER_YEAR / len(r)) * 100 - 100,
                        "excess_ann%": ex.mean() * PERIODS_PER_YEAR * 100, "excess_t": ex.mean() / (ex.std(ddof=1) / sqrt(len(ex))),
                        "Sharpe_ann": sharpe(net) * sqrt(PERIODS_PER_YEAR), "bench_Sharpe": sharpe(r.bench) * sqrt(PERIODS_PER_YEAR),
                        "maxDD%": max_drawdown(net) * 100, "bench_maxDD%": max_drawdown(r.bench) * 100, "DSR": dsr})
        print(pd.DataFrame(out).round(2).to_string(index=False))
        yr = (r.gross - r.bench).groupby(r.index.year).sum() * 100
        print("excess return by year (sum of quarterly, %):", {y: round(v, 1) for y, v in yr.items()})
        print(f"hit rate (quarters beating benchmark, gross): {(r.gross > r.bench).mean():.0%}")


if __name__ == "__main__":
    main()
