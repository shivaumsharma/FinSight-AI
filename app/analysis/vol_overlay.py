"""
vol_overlay.py

Volatility-targeting risk view (Moreira & Muir): hold less when recent volatility
is high, never more than 100%.

    suggested_exposure = min(1, target_vol / realised_vol)

Evidence (SPRINT_TRACKER.md, A2; S&P 500 2007-2025): max drawdown cut 39%
(2007-15) and 23% (2016-25), but no Sharpe gain in 2016-25. It is drawdown
protection, not a return forecast. Pure functions: the caller supplies prices.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.analysis.portfolio_metrics import max_drawdown

TARGET_VOL = 0.15      # annualised; round long-run equity volatility, not tuned to any backtest
WINDOW = 21            # trading days of realised volatility (about one month)
TRADING_DAYS = 252
MIN_OBSERVATIONS = 15  # fewer daily returns than this and a volatility estimate means nothing

EVIDENCE = {
    "market": "S&P 500 price index, 2007-2025, monthly rebalance, 5 bps cost",
    "target_vol": TARGET_VOL,
    "max_drawdown_cut": {"2007-2015": "-57% -> -35%", "2016-2025": "-34% -> -26%"},
    "sharpe": {"2007-2015": "0.28 -> 0.43", "2016-2025": "0.66 -> 0.68"},
    "takeaway": ("Reduced the worst peak-to-trough loss in both periods, but added no meaningful risk-adjusted "
                 "return in the 2016-2025 bull market. A risk-control tool, not a return forecast."),
}


def portfolio_value_series(prices: pd.DataFrame, quantities: Dict[str, float]) -> pd.Series:
    """Daily value of holding today's share counts: sum(price * quantity) over the holdings that have prices.
    Rows where any held ticker has no price yet are dropped, so a late-listing name cannot fake a jump."""
    cols = [t for t in prices.columns if t in quantities]
    if not cols:
        return pd.Series(dtype=float)
    sub = prices[cols].dropna(how="any")
    return (sub * pd.Series({t: quantities[t] for t in cols})).sum(axis=1)


def realised_volatility(returns: pd.Series, window: int = WINDOW) -> Optional[float]:
    """Annualised standard deviation of the last `window` daily returns, or None if there is too little data."""
    r = returns.dropna().tail(window)
    if len(r) < MIN_OBSERVATIONS:
        return None
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def suggested_exposure(vol: Optional[float], target_vol: float = TARGET_VOL, cap: float = 1.0) -> Optional[float]:
    """Fraction of the portfolio to keep invested to hold volatility near the target; never above `cap` (no leverage)."""
    if vol is None or vol <= 0:
        return None
    return float(min(cap, target_vol / vol))


def build_overlay(prices: pd.DataFrame, quantities: Dict[str, float]) -> Dict:
    """Risk-control summary for a set of holdings. `available` is False (with a reason) rather than raising when there
    is not enough price history."""
    value = portfolio_value_series(prices, quantities)
    if len(value) < MIN_OBSERVATIONS + 1:
        return {"available": False, "reason": "not enough overlapping price history for these holdings"}
    returns = value.pct_change().dropna()
    vol = realised_volatility(returns)
    if vol is None:
        return {"available": False, "reason": "not enough recent daily returns to estimate volatility"}
    exposure = suggested_exposure(vol)
    return {
        "available": True,
        "realised_volatility_pct": round(vol * 100, 1),
        "target_volatility_pct": round(TARGET_VOL * 100, 1),
        "suggested_exposure_pct": round(exposure * 100, 1),
        "suggested_cash_pct": round((1 - exposure) * 100, 1),
        "max_drawdown_in_window_pct": round(max_drawdown(value) * 100, 1),
        "window_days": int(len(value)),
        "evidence": EVIDENCE,
        "disclaimer": ("Illustrative risk control based on the last month of volatility. It describes how much risk "
                       "you are carrying, not what the market will do, and is not personalised investment advice."),
    }
