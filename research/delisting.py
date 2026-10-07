"""
delisting.py -- RESEARCH SANDBOX. Shared handling for stocks that were delisted (acquired / bankrupt / merged away).

Two rules keep the survivorship fix honest:
  1. TERMINAL RETURN: in the adjusted-close matrix, a delisted ticker's last real price is carried forward (as if sold at
     that price). A quarter in which the company was acquired then shows the real return up to the deal, instead of NaN --
     NaN would silently drop the name and bring the survivorship bias straight back.
  2. NO STALE FEATURES: after the last real price, price-based features and scores must NOT be computed from the carried
     price. `stale_tickers(date, tickers)` marks those names so callers can mask them.

`research/data/delisting_last_dates.json` maps ticker -> last real trading date; it is written by delisted_build.py.
"""
import json
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent / "data" / "delisting_last_dates.json"
GRACE_DAYS = 7  # a price this recent still counts as "live" on the evaluation date


def load():
    if not PATH.exists():
        return {}
    return {t: pd.Timestamp(d) for t, d in json.loads(PATH.read_text()).items()}


def stale_tickers(date, tickers, last_real=None):
    """Boolean Series over `tickers`: True where `date` is more than GRACE_DAYS after the ticker's last real price."""
    last_real = load() if last_real is None else last_real
    flags = {t: (t in last_real and pd.Timestamp(date) > last_real[t] + pd.Timedelta(days=GRACE_DAYS)) for t in tickers}
    return pd.Series(flags, index=list(tickers))
