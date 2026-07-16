"""
institutional_ratings.py

Retrieves real, currently-covering analyst firms' latest ratings for a
ticker via yfinance's upgrades_downgrades feed, and normalizes each
firm's inconsistent rating vocabulary into FinSight's own canonical
BUY/HOLD/SELL scale.

There is deliberately no fixed list of "the institutions we track" --
some well-known institutions (e.g. asset managers like BlackRock, or
ratings services with their own proprietary scale like Morningstar)
don't publish sell-side BUY/HOLD/SELL calls in this feed at all, so
hardcoding specific names would mean silently faking coverage that
doesn't exist. This uses whichever real firms actually cover the
ticker -- most recent rating per firm, capped at MAX_INSTITUTIONS so
a heavily-covered mega-cap doesn't produce an unreadably long list.
"""

import yfinance as yf

MAX_INSTITUTIONS = 10

# Sell-side firms use wildly inconsistent vocabulary for the same
# underlying call (e.g. "Overweight"/"Outperform"/"Buy" all mean
# bullish). Deterministic, explicit mapping -- not an LLM -- same
# reasoning as ticker resolution: this is a lookup problem against a
# small, enumerable vocabulary, not a reasoning problem.
_GRADE_MAP = {
    "buy": "BUY", "strong buy": "BUY", "overweight": "BUY",
    "outperform": "BUY", "positive": "BUY", "accumulate": "BUY",
    "add": "BUY",
    "hold": "HOLD", "neutral": "HOLD", "sector weight": "HOLD",
    "perform": "HOLD", "market perform": "HOLD", "equal-weight": "HOLD",
    "peer perform": "HOLD", "in-line": "HOLD", "sector perform": "HOLD",
    "sell": "SELL", "underperform": "SELL", "underweight": "SELL",
    "reduce": "SELL", "negative": "SELL", "strong sell": "SELL",
}


def _normalize_grade(raw_grade: str):
    return _GRADE_MAP.get(raw_grade.strip().lower())


def fetch_institutional_ratings(ticker: str):
    """
    Returns a list of {"firm", "rating" ("BUY"/"HOLD"/"SELL"),
    "raw_grade", "date"} for the most recent rating from each real
    firm currently covering `ticker`, most recent first, capped at
    MAX_INSTITUTIONS. Returns [] if yfinance has no upgrades/
    downgrades data for this ticker (not every stock has analyst
    coverage) or the ticker/network lookup fails -- callers should
    treat that as "no consensus data available", not an error.
    """
    try:
        df = yf.Ticker(ticker).upgrades_downgrades
    except Exception:
        return []

    if df is None or df.empty:
        return []

    df = df.sort_index(ascending=False)

    seen_firms = set()
    ratings = []

    for date, row in df.iterrows():
        firm = row.get("Firm")
        if not firm or firm in seen_firms:
            continue

        normalized = _normalize_grade(str(row.get("ToGrade", "")))
        if normalized is None:
            # Unrecognized grade vocabulary -- skip rather than guess
            # at a classification we can't stand behind.
            continue

        seen_firms.add(firm)
        ratings.append({
            "firm": firm,
            "rating": normalized,
            "raw_grade": row.get("ToGrade"),
            "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
        })

        if len(ratings) >= MAX_INSTITUTIONS:
            break

    return ratings
