"""
portfolio_metrics.py

Hand-rolled portfolio-level performance/risk metrics -- CAGR,
annualized volatility, Sharpe, Sortino, max drawdown, hit rate, profit
factor, turnover -- for scripts/walkforward_backtest.py's equity
curve. No new dependency (empyrical/quantstats/pyfolio checked, none
currently installed -- see requirements.txt) -- same "hand-roll a
well-known formula rather than pull in a library" precedent as
alpha_factors.py's own _annualized_volatility(), just generalized here
to an arbitrary portfolio-level series instead of one ticker's
trailing 12 months.

Every function is a pure function over a plain pandas Series -- no
FinSight-specific types, no I/O -- so the walk-forward runner (or any
future caller) can build an equity curve however it likes and pass it
straight in.

Conventions used throughout:
- `equity_curve`: a pd.Series of portfolio VALUE (not returns), a
  DatetimeIndex, strictly increasing index, first value is the
  starting capital.
- `returns` / `period_returns`: a pd.Series of PERIODIC (not
  daily-annualized) simple returns, e.g. one value per rebalance
  period -- the ratio functions below annualize internally, given how
  many periods occur per year (`periods_per_year`).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def periodic_returns(equity_curve: pd.Series) -> pd.Series:
    """Simple period-over-period returns from a value curve -- the
    shared building block every metric below is computed from, so
    there's exactly one definition of "return" in this module."""
    return equity_curve.pct_change().dropna()


def cagr(equity_curve: pd.Series) -> Optional[float]:
    """Compound annual growth rate from the first to the last value in
    equity_curve, using the ACTUAL elapsed calendar time between them
    (not periods_per_year x number-of-periods, which would be wrong
    for an irregular or partial-year curve) -- requires a
    DatetimeIndex. None if the curve has fewer than 2 points, spans
    zero calendar time, or starts at a non-positive value (a CAGR is
    meaningless against a starting value of 0 or less)."""
    if equity_curve is None or len(equity_curve) < 2:
        return None
    start_value = float(equity_curve.iloc[0])
    end_value = float(equity_curve.iloc[-1])
    if start_value <= 0:
        return None
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> Optional[float]:
    """Standard deviation of periodic returns, annualized by the
    sqrt-of-time rule -- same formula as alpha_factors.py's
    _annualized_volatility(), generalized to any periods_per_year
    (that one is hardcoded to daily/252; a walk-forward equity curve's
    natural period is one rebalance -- e.g. 4/year for quarterly, not
    252)."""
    if returns is None or len(returns) < 2:
        return None
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> Optional[float]:
    """Annualized Sharpe: (mean excess periodic return / periodic
    return stdev) x sqrt(periods_per_year) -- the standard textbook
    form, not a rolling or log-return variant. risk_free_rate is an
    ANNUAL rate (e.g. the ^TNX yield already used elsewhere in this
    project, see phase2_backtest.py's tnx_history) -- converted to a
    per-period rate here so it's subtracted on the same footing as
    `returns`, not accidentally compared annual-vs-periodic. None (not
    0) if returns has zero variance -- an undefined ratio, not a real
    zero-risk result."""
    if returns is None or len(returns) < 2:
        return None
    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - period_rf
    std = excess.std()
    if not std:
        return None
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> Optional[float]:
    """Same shape as sharpe_ratio, but the denominator is downside
    deviation only (root-mean-square of the negative-excess-return
    periods, positive-excess periods zeroed rather than dropped --
    the classic Sortino/Rom definition, which keeps N as the full
    sample size) -- a strategy with large upside swings and small/no
    downside swings scores better here than under Sharpe, which
    penalizes both equally. None (not 0 or inf) if there are no
    downside periods at all in the sample -- an undefined ratio, not
    literally "infinite skill"."""
    if returns is None or len(returns) < 2:
        return None
    period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - period_rf
    downside = excess.clip(upper=0)
    downside_std = np.sqrt((downside ** 2).mean())
    if not downside_std:
        return None
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> Optional[float]:
    """Largest peak-to-trough decline in equity_curve, as a negative
    fraction (-0.23 = a 23% drawdown) -- the worst point-to-point loss
    an investor holding this exact path would have experienced,
    regardless of the final value."""
    if equity_curve is None or len(equity_curve) < 2:
        return None
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def hit_rate(period_returns: pd.Series) -> Optional[float]:
    """Fraction of periods with a strictly positive return -- the
    portfolio-level analog of Buy/Sell precision elsewhere in this
    project (EVALUATION.md section 0), just at the whole-strategy
    level instead of per-call."""
    if period_returns is None or len(period_returns) == 0:
        return None
    return float((period_returns > 0).mean())


def profit_factor(period_returns: pd.Series) -> Optional[float]:
    """Gross gains / gross losses across all periods -- >1 means
    winning periods outweighed losing ones in total magnitude, not
    just in count (unlike hit_rate). None if there were no losing
    periods at all (undefined, not infinite)."""
    if period_returns is None or len(period_returns) == 0:
        return None
    gains = period_returns[period_returns > 0].sum()
    losses = -period_returns[period_returns < 0].sum()
    if not losses:
        return None
    return float(gains / losses)


def turnover(previous_weights: Dict[str, float], target_weights: Dict[str, float]) -> float:
    """One-sided turnover -- half the sum of absolute weight changes
    across every ticker that appears in either period (a ticker
    dropped entirely counts as a change from its old weight to 0, a
    newly-added one from 0 to its new weight). The standard
    "buy-side-only" turnover definition: fully replacing the portfolio
    is turnover=1.0, not 2.0 (which the unhalved sum of |changes|
    would give, double-counting each position's sell and the
    replacement's buy)."""
    tickers = set(previous_weights) | set(target_weights)
    total_change = sum(abs(target_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) for t in tickers)
    return total_change / 2.0
