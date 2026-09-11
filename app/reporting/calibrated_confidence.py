"""
calibrated_confidence.py

build_calibrated_confidence(sector, rating) -- "how often has FinSight's
OWN historical Buy/Hold/Sell calls in THIS sector actually been right,"
read from the exact same two backtest files scripts/canonical_accuracy.py
pools (see that module's own docstring for why these files and not the
live call tracker: this deploys to hosts with an ephemeral filesystem,
so only git-committed artifacts survive a redeploy).

Falls back from (sector, rating) -> (rating only, all sectors) -> None
whenever a bucket doesn't have enough samples to mean anything -- a
"58% accurate" claim from n=3 is noise dressed as a number, not a real
calibration.
"""

import json
from pathlib import Path
from typing import Optional

from app.analysis.baseline_scoring import score_rating

_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"

_SOURCE_FILES = [
    "backtest_results_ticker_universe_asof12mo_exit0mo.json",
    "backtest_results_ticker_universe_asof24mo_exit12mo.json",
]

# Below this, a bucket's accuracy is not shown at all -- see this
# module's own docstring for why. 20 is a judgment call, not a tuned
# statistical threshold: small enough that most sector/rating
# combinations in a 1,900-row pooled backtest actually clear it, large
# enough that a handful of lucky/unlucky calls can't swing the number.
MIN_SAMPLE_SIZE = 20

# The backtest's own `category` field is GICS sector names (S&P index
# membership data, e.g. "Information Technology (S&P 500)") -- but
# company_info's own "sector" field (app/data/market_data.py) is
# straight from yfinance's `.info["sector"]`, which uses a DIFFERENT,
# non-GICS taxonomy for several sectors. Confirmed live: without this
# map, an exact string match against a real ticker's sector NEVER hits
# (e.g. yfinance's "Technology" never equals the backtest's "Information
# Technology"), so every report would silently fall back to the
# less-specific "all sectors" bucket -- no crash, no error, just a
# strictly worse number shown as if it were the best available one.
_YFINANCE_SECTOR_TO_GICS = {
    "Technology": "Information Technology",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
    # Already GICS-named in yfinance's own taxonomy -- listed for
    # completeness/clarity, not because they'd otherwise fail to match.
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}


def _sector_from_category(category: Optional[str]) -> Optional[str]:
    """category looks like "Industrials (S&P 500)" -- strip the
    index-membership suffix to get the bare GICS sector name."""
    if not category:
        return None
    return category.split(" (")[0].strip()


def _load_pooled_rows() -> list:
    rows = []
    for filename in _SOURCE_FILES:
        path = _SCRIPT_DIR / filename
        try:
            with open(path) as f:
                rows.extend(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    # Same exclusion score_rating() itself already applies -- see
    # canonical_accuracy.py's load_source(), mirrored here rather than
    # imported since that function is file-path-bound to one filename
    # at a time, not the pooling this needs.
    return [
        r for r in rows
        if r.get("recommendation") not in (None, "Insufficient Data")
        and r.get("realized_return_pct") is not None
    ]


def _accuracy_for(rows: list, rating: str, gics_sector: Optional[str]) -> Optional[dict]:
    matching = [
        r for r in rows
        if r["recommendation"] == rating
        and (gics_sector is None or _sector_from_category(r.get("category")) == gics_sector)
    ]
    n = len(matching)
    if n < MIN_SAMPLE_SIZE:
        return None
    correct = sum(1 for r in matching if score_rating(r["recommendation"], r["realized_return_pct"]))
    return {"accuracy_pct": round(100 * correct / n, 1), "n": n}


def build_calibrated_confidence(sector: Optional[str], rating: str) -> Optional[dict]:
    """sector: company_overview's own "sector" field (yfinance
    taxonomy, e.g. "Technology"), rating: the report's own Buy/Hold/
    Sell call. None whenever no bucket (sector-level or overall) has
    enough samples, or rating isn't a real call ("Insufficient Data"
    included)."""
    if rating not in ("Buy", "Hold", "Sell"):
        return None

    rows = _load_pooled_rows()
    if not rows:
        return None

    gics_sector = _YFINANCE_SECTOR_TO_GICS.get(sector) if sector else None
    if gics_sector:
        sector_level = _accuracy_for(rows, rating, gics_sector)
        if sector_level:
            return {**sector_level, "scope": "sector", "sector": gics_sector, "rating": rating}

    overall_level = _accuracy_for(rows, rating, None)
    if overall_level:
        return {**overall_level, "scope": "overall", "sector": None, "rating": rating}

    return None
