"""
peer_comparison.py

Powers the stock-detail-page's Peer Comparison table -- a user-entered
peer ticker (never auto-discovered; this app has no peer-multiple/
industry data source, see relative_valuation.py's own docstring)
compared side by side on Market Cap / P/E / Revenue CAGR / EPS CAGR.

Deliberately reuses the already-built, already-tested Phase 1/2
functions (get_stock_overview, build_growth_metrics) rather than
re-fetching or re-deriving anything -- both are cheap enough to call
twice (once per ticker) for an on-demand comparison table.

ROE/ROCE are NOT included here (unlike the plan's original wishlist) --
they're already shown for the primary ticker in FundamentalsCard/
PerformanceDashboard, and adding a second full financial-statement
fetch pass here just to re-derive them for the peer wasn't worth the
extra latency for a comparison table whose real point is market cap/
valuation/growth at a glance.
"""

from app.analysis.growth_metrics import build_growth_metrics
from app.data.stock_overview import get_stock_overview


def _peer_row(ticker: str) -> dict:
    overview = get_stock_overview(ticker)
    growth = build_growth_metrics(ticker)
    return {
        "ticker": overview["ticker"],
        "company_name": overview["company_name"],
        "currency": overview["currency"],
        "market_cap": overview["market_cap"],
        "trailing_pe": overview["fundamentals"]["trailing_pe"],
        "revenue_cagr_3y": growth["revenue_cagr"]["3y"],
        "eps_cagr_3y": growth["eps_cagr"]["3y"],
    }


def build_peer_comparison(ticker: str, peer_ticker: str) -> dict:
    """Raises TickerNotFoundError (via get_stock_overview) if either
    ticker is invalid -- callers should surface which one failed is
    not distinguishable here, so main.py's endpoint reports the
    generic "ticker not found" for whichever one it was."""
    return {"primary": _peer_row(ticker), "peer": _peer_row(peer_ticker)}
