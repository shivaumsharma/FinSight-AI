"""
risk_range.py

A plausible range for the stock's price over the next quarter and year, from its own trailing volatility. It
says how far the price could move, not which way; the Buy/Hold/Sell rating says nothing about size.
The 10th-90th percentile band is checked against realised outcomes in research/risk_range_calibration.py.
"""

import math
from typing import Any, Dict, Optional

import pandas as pd

TRADING_DAYS = 252
WINDOW = 252          # one year of daily returns for the volatility estimate
MIN_OBSERVATIONS = 120
Z_80 = 1.2816         # 10th/90th percentile of a standard normal
HORIZONS = {"1 quarter": 63, "1 year": 252}


def annualised_volatility(close: pd.Series) -> Optional[float]:
    returns = close.dropna().pct_change().dropna().tail(WINDOW)
    if len(returns) < MIN_OBSERVATIONS:
        return None
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def price_range(price: float, vol: float, days: int) -> Dict[str, float]:
    """10th-90th percentile price after `days` trading days; log-normal, drift ignored (small against the spread)."""
    sd = vol * math.sqrt(days / TRADING_DAYS)
    return {"low": round(price * math.exp(-Z_80 * sd), 2), "high": round(price * math.exp(Z_80 * sd), 2),
            "low_pct": round((math.exp(-Z_80 * sd) - 1) * 100, 1), "high_pct": round((math.exp(Z_80 * sd) - 1) * 100, 1)}


def build_risk_range(historical_prices: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if historical_prices is None or getattr(historical_prices, "empty", True) or "Close" not in historical_prices.columns:
        return None
    close = historical_prices["Close"].dropna()
    vol = annualised_volatility(close)
    if vol is None or vol <= 0:
        return None
    price = float(close.iloc[-1])
    return {"annualised_volatility_pct": round(vol * 100, 1), "current_price": round(price, 2),
            "coverage": "80% of outcomes",
            "ranges": {name: price_range(price, vol, days) for name, days in HORIZONS.items()},
            "note": "Range from the last year's volatility; it shows size of move, not direction, and a large "
                    "shock can fall outside it."}
