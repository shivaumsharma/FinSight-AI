"""
growth_metrics.py

Powers the stock-detail-page's Financial Performance chart (quarterly/
yearly revenue, net income, margins) and Growth Metrics tiles (1/3/5-year
CAGR for Revenue, EPS, Book Value, FCF) -- see GET /v1/stocks/{ticker}/
financials in app/api/main.py.

Reuses FinancialStatementNormaliser (app/data/financial_normalizer.py)
against BOTH MarketDataLoader's annual and quarterly statement fetches
(the normalizer itself is period-agnostic -- it just aligns/cleans
whatever income/balance/cash-flow DataFrames it's given), rather than
duplicating any statement-parsing logic here.

CAGR is deliberately computed from ANNUAL data only, even when the
caller is viewing the quarterly chart -- multi-year growth on 4
quarterly points is noise, not signal. Each of the 1/3/5-year buckets
is independently gated on having enough annual history: yfinance's
default annual financials typically return ~4 fiscal years, so a 5-year
CAGR is honestly "Unavailable" far more often than not -- this returns
None rather than fabricating a number from too little data, same
"Unavailable" convention app/analysis/financial_analysis.py's own
Revenue CAGR already uses.
"""

from typing import Optional

import pandas as pd

from app.data.crypto_resolver import is_crypto_ticker
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.data.market_data import MarketDataLoader

CAGR_YEAR_BUCKETS = (1, 3, 5)


def _compute_cagr(series: pd.Series, years: int) -> Optional[float]:
    """CAGR (%) over the most recent `years` annual periods, using the
    latest value and the value `years` periods before it -- None if
    there isn't enough history, either endpoint is non-positive (a CAGR
    on a loss-making base year is not a meaningful percentage), or the
    starting value is zero."""
    clean = series.dropna()
    if len(clean) < years + 1:
        return None
    start = clean.iloc[-(years + 1)]
    end = clean.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    return round(((end / start) ** (1 / years) - 1) * 100, 2)


def _per_share(numerator: pd.Series, shares: pd.Series) -> pd.Series:
    """numerator / shares_outstanding, aligned on the shared index --
    used for EPS (net_income/shares) and Book Value per share
    (total_equity/shares). Rows where either side is missing drop out
    rather than producing a spurious inf/NaN entry."""
    aligned = pd.concat([numerator, shares], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0] / aligned.iloc[:, 1]


def build_growth_metrics(ticker: str) -> dict:
    """{"revenue_cagr": {"1y": .., "3y": .., "5y": ..}, "eps_cagr": {...},
    "book_value_cagr": {...}, "fcf_cagr": {...}} -- every leaf is a
    percent (float) or None. Always computed from annual statements."""
    if is_crypto_ticker(ticker):
        # No income/balance/cash-flow statements exist for crypto --
        # MarketDataLoader's own statement fetch raises on an empty
        # DataFrame (see market_data_tool.py's crypto guard for the
        # same finding), so this returns the same all-None shape
        # up front rather than letting that raise reach the endpoint.
        empty = {f"{y}y": None for y in CAGR_YEAR_BUCKETS}
        return {"revenue_cagr": dict(empty), "eps_cagr": dict(empty), "book_value_cagr": dict(empty), "fcf_cagr": dict(empty)}

    loader = MarketDataLoader(ticker)
    normalizer = FinancialStatementNormaliser(
        loader.get_income_statement(), loader.get_balance_sheet(), loader.get_cash_flow()
    )
    df = normalizer.normalise()

    eps = _per_share(df.get("net_income", pd.Series(dtype=float)), df.get("shares_outstanding", pd.Series(dtype=float)))
    book_value_per_share = _per_share(
        df.get("total_equity", pd.Series(dtype=float)), df.get("shares_outstanding", pd.Series(dtype=float))
    )
    fcf = (df.get("cash_from_operations", pd.Series(dtype=float)) - df.get("capex", pd.Series(dtype=float))).dropna()

    def _bucket(series: pd.Series) -> dict:
        return {f"{y}y": _compute_cagr(series, y) for y in CAGR_YEAR_BUCKETS}

    return {
        "revenue_cagr": _bucket(df.get("revenue", pd.Series(dtype=float))),
        "eps_cagr": _bucket(eps),
        "book_value_cagr": _bucket(book_value_per_share),
        "fcf_cagr": _bucket(fcf),
    }


def build_financial_performance(ticker: str, period: str = "yearly") -> list:
    """Time-series rows for the Financial Performance chart, oldest
    first: [{"period_end": "2024-09-30", "revenue": .., "net_income": ..,
    "ebit": .., "net_margin_pct": .., "operating_margin_pct": ..}, ...].
    `period` is "yearly" (default) or "quarterly". Margin fields are
    None wherever revenue is zero/missing for that period (never a
    divide-by-zero crash, never a fabricated 0%)."""
    if is_crypto_ticker(ticker):
        return []

    loader = MarketDataLoader(ticker)
    if period == "quarterly":
        income, balance, cash_flow = (
            loader.get_quarterly_income_statement(), loader.get_quarterly_balance_sheet(), loader.get_quarterly_cash_flow(),
        )
    else:
        income, balance, cash_flow = loader.get_income_statement(), loader.get_balance_sheet(), loader.get_cash_flow()

    df = FinancialStatementNormaliser(income, balance, cash_flow).normalise()

    rows = []
    for period_end, row in df.iterrows():
        revenue = row.get("revenue")
        net_income = row.get("net_income")
        ebit = row.get("ebit")
        net_margin = round(net_income / revenue * 100, 2) if pd.notna(revenue) and revenue and pd.notna(net_income) else None
        operating_margin = round(ebit / revenue * 100, 2) if pd.notna(revenue) and revenue and pd.notna(ebit) else None
        rows.append({
            "period_end": period_end.date().isoformat(),
            "revenue": None if pd.isna(revenue) else float(revenue),
            "net_income": None if pd.isna(net_income) else float(net_income),
            "ebit": None if pd.isna(ebit) else float(ebit),
            "net_margin_pct": net_margin,
            "operating_margin_pct": operating_margin,
        })
    return rows
