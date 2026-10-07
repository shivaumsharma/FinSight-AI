"""
portfolio_risk_overlay.py

build_risk_overlay(user_id) is GET /v1/portfolio/risk-overlay's body: the user's own self-reported holdings run through
app/analysis/vol_overlay.py. Same extraction pattern as portfolio_summary.py's build_portfolio_view().

Degrades instead of failing: no holdings, no usable prices, or a throttled price fetch all return
{"available": False, "reason": ...} so the endpoint never 500s a portfolio page. Non-USD holdings (NSE ".NS") are left
out and listed, since mixing currencies in one value series would be meaningless.
"""

from app.analysis import vol_overlay
from app.api import db

HISTORY_PERIOD = "6mo"


def _download_prices(tickers):
    import yfinance as yf

    data = yf.download(list(tickers), period=HISTORY_PERIOD, auto_adjust=True, progress=False)["Close"]
    if getattr(data, "ndim", 2) == 1:  # a single ticker comes back as a Series
        data = data.to_frame(name=list(tickers)[0])
    return data


def build_risk_overlay(user_id: str) -> dict:
    holdings = db.get_portfolio_holdings(user_id)
    if not holdings:
        return {"available": False, "reason": "no holdings in the portfolio yet"}
    usd = {h["ticker"]: float(h["quantity"]) for h in holdings if not h["ticker"].upper().endswith(".NS")}
    excluded = sorted(h["ticker"] for h in holdings if h["ticker"].upper().endswith(".NS"))
    if not usd:
        return {"available": False, "reason": "only non-USD holdings; the overlay covers USD-priced holdings",
                "excluded_tickers": excluded}
    try:
        prices = _download_prices(usd)
    except Exception:
        return {"available": False, "reason": "price history is temporarily unavailable", "excluded_tickers": excluded}
    result = vol_overlay.build_overlay(prices, usd)
    result["excluded_tickers"] = excluded
    result["holdings_used"] = sorted(t for t in usd if t in getattr(prices, "columns", []))
    return result
