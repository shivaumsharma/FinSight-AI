"""
rlvr_context.py

Point-in-time context construction for RLVR training-data generation --
reuses scripts/phase2_backtest.py's own no-look-ahead methodology
(_tz_naive/_price_on_or_before/_trailing_beta/_point_in_time_statement,
see that module's docstring) rather than re-deriving it, since the
reward this pipeline trains against is only meaningful if it's built the
exact same way the project's own trusted backtest already measures
"correct."

Deliberately split from scripts/phase2_backtest.py's run_one() at the
point where that function calls derive_recommendation() (app/reporting/
report_data_builder.py) -- this module runs ValuationTool the same way
(so the RL prompt can cite real revenue/margin/DCF/relative-valuation
facts) but stops short of the deterministic rule-based rating, since
handing the model that rating in its prompt would leak the answer it's
supposed to be learning to produce.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from phase2_backtest import (  # noqa: E402
    _point_in_time_statement,
    _price_on_or_before,
    _trailing_beta,
    _tz_naive,
)

from app.core.research_context import ResearchContext
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.tools.valuation_tool import ValuationTool


class PointInTimeUnavailable(Exception):
    """Raised when this ticker/as_of_date combination can't produce a
    valid point-in-time context -- same set of reasons run_one() raises
    ValueError for (missing price history, missing financials, no fiscal
    year filed early enough, no shares outstanding). Callers (the
    dataset builder) should skip and move on, same as phase2_backtest.py
    already does for these cases."""


def build_point_in_time_context(
    ticker: str, as_of_date, today_date, market_history
) -> Tuple[ResearchContext, float]:
    """
    Reconstructs what `ticker` looked like as of `as_of_date` (no
    look-ahead) and runs the real valuation pipeline against it, exactly
    as scripts/phase2_backtest.py's run_one() does -- but returns the
    populated ResearchContext and the realized return over
    [as_of_date, today_date] instead of a derived rating, so the caller
    can build a training prompt and use the realized return as reward
    ground truth without the rule-based rating ever entering the prompt.
    """
    stock = yf.Ticker(ticker)

    price_history = _tz_naive(stock.history(period="10y"))
    if price_history is None or price_history.empty:
        raise PointInTimeUnavailable(f"{ticker}: no price history available")

    price_as_of = _price_on_or_before(price_history, as_of_date)
    price_today = _price_on_or_before(price_history, today_date)
    if price_as_of is None or price_today is None:
        raise PointInTimeUnavailable(
            f"{ticker}: insufficient price history spanning as-of date and today"
        )
    # _price_on_or_before can return a real NaN (not None) if yfinance's
    # own Close column has a NaN row on-or-before the target date --
    # found live: every example in a full dataset build silently got a
    # NaN realized_return_pct this way, which downstream score_rating()
    # treats as an ordinary (non-None) False rather than "unscored,"
    # so the GRPO reward for an entire training run was silently
    # guaranteed 0.0 regardless of what the model predicted. Raising
    # here instead means build_rlvr_dataset.py's existing skip-and-log
    # path catches it, the same as any other bad-data case.
    if price_as_of != price_as_of or price_today != price_today:
        raise PointInTimeUnavailable(f"{ticker}: NaN price on-or-before as-of date or today")

    realized_return_pct = (price_today - price_as_of) / price_as_of * 100

    beta = _trailing_beta(price_history, market_history, as_of_date) or 1.2

    income = stock.financials
    balance = stock.balance_sheet
    cashflow = stock.cashflow
    if income.empty or balance.empty or cashflow.empty:
        raise PointInTimeUnavailable(f"{ticker}: financial statements unavailable")

    income_pit = _point_in_time_statement(income, as_of_date)
    balance_pit = _point_in_time_statement(balance, as_of_date)
    cashflow_pit = _point_in_time_statement(cashflow, as_of_date)
    if income_pit.empty or balance_pit.empty or cashflow_pit.empty:
        raise PointInTimeUnavailable(
            f"{ticker}: no fiscal year was filed early enough to be known as of the as-of date"
        )

    financial_df = FinancialStatementNormaliser(income_pit, balance_pit, cashflow_pit).normalise()
    if financial_df.empty or len(financial_df) < 2:
        raise PointInTimeUnavailable(
            f"{ticker}: insufficient point-in-time financial history (need >=2 fiscal years)"
        )

    shares_outstanding = None
    if "shares_outstanding" in financial_df.columns:
        series = financial_df["shares_outstanding"].dropna()
        if not series.empty:
            shares_outstanding = series.iloc[-1]
    if not shares_outstanding:
        raise PointInTimeUnavailable(f"{ticker}: no point-in-time shares outstanding")

    market_cap_as_of = price_as_of * shares_outstanding

    ctx = ResearchContext(
        ticker=ticker,
        question=f"Should I buy {ticker}? (RLVR training context as-of {as_of_date.date()})",
    )
    ctx.normalized_financials = financial_df
    ctx.market_cap = market_cap_as_of
    ctx.beta = beta
    ctx.historical_prices = price_history[price_history.index <= as_of_date]
    ctx.company_info = {
        "current_price": price_as_of,
        "market_cap": market_cap_as_of,
        "beta": beta,
    }

    ValuationTool().run(ctx)

    return ctx, realized_return_pct


def fetch_market_history(period: str = "5y"):
    """Same benchmark series phase2_backtest.py's main() fetches once
    per run -- factored out here so the dataset builder can fetch it
    once for a whole batch of tickers/dates, not per-ticker."""
    from phase2_backtest import MARKET_BENCHMARK  # noqa: E402

    return _tz_naive(yf.Ticker(MARKET_BENCHMARK).history(period=period))
