"""
wacc_capm_audit.py

Diagnostic-only, read-only report: for every ticker in the same
point-in-time backtest universe/window phase2_backtest.py already
uses, surfaces the actual WACC/CAPM inputs (computed WACC, raw WACC
before any floor, beta, risk-free rate, equity risk premium, terminal
growth) and intrinsic-value-to-price ratio that produced that
ticker's Buy/Hold/Sell call, plus the distribution of that ratio
across the universe. Does not import or touch anything in
app/valuation/ beyond calling ValuationTool exactly as
phase2_backtest.py already does -- no valuation logic is changed by
this file, it only reports what the existing logic actually computed.

Reuses phase2_backtest.py's own point-in-time helpers (_tz_naive,
_price_on_or_before, _trailing_beta, _point_in_time_statement, TICKERS,
MARKET_BENCHMARK) rather than duplicating that logic, so this is
diagnosing the EXACT SAME per-ticker WACC/intrinsic-value calculations
that produced the Buy/Hold/Sell calls scripts/phase2_backtest.py's own
accuracy numbers are based on -- not a separate, potentially-diverging
recomputation.

Why this exists: scripts/backtest results show the model's Sell calls
losing badly to a naive Always-Buy baseline in both backtest windows.
One candidate explanation is that the recommendation logic itself is
miscalibrated (thresholds, weights); another is that the DCF's inputs
are systematically biased low regardless of the company being valued,
which would make Sell calls an artifact of the assumptions, not a
real read on any specific company. valuation_pipeline.py's own module
docstring already documents finding and partially correcting exactly
this failure mode once before (median DCF-implied upside was -30.8%
across a 1,000-ticker universe with a 3% terminal growth assumption;
raised to 4% as "the first, cheapest candidate fix" -- explicitly
flagged as unverified whether that alone closed the gap). This script
re-runs that same check with current data across all the WACC/CAPM
inputs, not just terminal growth.

Usage:
    python scripts/wacc_capm_audit.py [as_of_months_ago] [exit_months_ago]
    (same positional args as phase2_backtest.py; default 12 0)
"""

import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from app.core.research_context import ResearchContext
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.tools.valuation_tool import ValuationTool
from app.valuation.wacc_engine import WACCEngine

from phase2_backtest import (
    TICKERS, MARKET_BENCHMARK,
    _tz_naive, _price_on_or_before, _trailing_beta, _point_in_time_statement,
)

# WACCEngine's own hardcoded defaults (app/valuation/wacc_engine.py,
# __init__ signature) -- not fetched from any live data source (no
# Treasury-yield API, no dynamic equity-risk-premium estimate), not
# overridden by any caller (valuation_pipeline.py constructs
# WACCEngine with only financial_df/market_cap/beta -- confirmed by
# reading every WACCEngine(...) call site in app/). Same two numbers
# for every ticker, every point in time, regardless of actual macro
# conditions at the as-of date.
RISK_FREE_RATE_SOURCE = WACCEngine.__init__.__defaults__[0]
MARKET_RISK_PREMIUM_SOURCE = WACCEngine.__init__.__defaults__[1]


def audit_one(ticker, category, as_of_date, market_history):
    stock = yf.Ticker(ticker)
    price_history = _tz_naive(stock.history(period="10y"))
    if price_history is None or price_history.empty:
        raise ValueError("no price history available")

    price_as_of = _price_on_or_before(price_history, as_of_date)
    if price_as_of is None:
        raise ValueError("no price as of the as-of date")

    beta = _trailing_beta(price_history, market_history, as_of_date) or 1.2

    income = stock.financials
    balance = stock.balance_sheet
    cashflow = stock.cashflow
    if income.empty or balance.empty or cashflow.empty:
        raise ValueError("financial statements unavailable")

    income_pit = _point_in_time_statement(income, as_of_date)
    balance_pit = _point_in_time_statement(balance, as_of_date)
    cashflow_pit = _point_in_time_statement(cashflow, as_of_date)
    if income_pit.empty or balance_pit.empty or cashflow_pit.empty:
        raise ValueError("no fiscal year known as of the as-of date")

    financial_df = FinancialStatementNormaliser(income_pit, balance_pit, cashflow_pit).normalise()
    if financial_df.empty or len(financial_df) < 2:
        raise ValueError("insufficient point-in-time financial history")

    shares_outstanding = None
    if "shares_outstanding" in financial_df.columns:
        series = financial_df["shares_outstanding"].dropna()
        if not series.empty:
            shares_outstanding = series.iloc[-1]
    if not shares_outstanding:
        raise ValueError("no point-in-time shares outstanding")

    market_cap_as_of = price_as_of * shares_outstanding

    ctx = ResearchContext(ticker=ticker, question=f"WACC/CAPM audit for {ticker}")
    ctx.normalized_financials = financial_df
    ctx.market_cap = market_cap_as_of
    ctx.beta = beta
    ctx.historical_prices = price_history[price_history.index <= as_of_date]
    ctx.company_info = {"current_price": price_as_of, "market_cap": market_cap_as_of, "beta": beta}

    ValuationTool().run(ctx)
    vr = ctx.valuation_results

    intrinsic_value = vr.get("intrinsic_value")
    iv_to_price = (intrinsic_value / price_as_of) if intrinsic_value and price_as_of else None

    return {
        "ticker": ticker,
        "category": category,
        "dcf_available": vr.get("dcf_available"),
        "dcf_unavailable_reason": vr.get("dcf_unavailable_reason"),
        "beta": beta,
        "risk_free_rate": RISK_FREE_RATE_SOURCE,
        "equity_risk_premium": MARKET_RISK_PREMIUM_SOURCE,
        "cost_of_equity": (RISK_FREE_RATE_SOURCE + beta * MARKET_RISK_PREMIUM_SOURCE) if beta is not None else None,
        "raw_wacc": vr.get("raw_wacc"),
        "wacc_used": vr.get("wacc"),
        "wacc_floored": vr.get("wacc_floored"),
        "terminal_growth_rate": vr.get("terminal_growth_rate"),
        "price_as_of": price_as_of,
        "intrinsic_value": intrinsic_value,
        "iv_to_price": iv_to_price,
        "error": None,
    }


def main():
    as_of_months_ago = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    exit_months_ago = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # noqa: unused -- accepted for CLI-arg parity with phase2_backtest.py, not needed here (no realized return computed)

    today_date = pd.Timestamp(datetime.utcnow().date())
    as_of_date = today_date - pd.Timedelta(days=as_of_months_ago * 30)

    print(f"Universe: curated ({len(TICKERS)} tickers)   As-of date: {as_of_date.date()}", file=sys.stderr)
    print(f"CAPM inputs (app/valuation/wacc_engine.py hardcoded defaults, not live-fetched, "
          f"same for every ticker/date): risk_free_rate={RISK_FREE_RATE_SOURCE:.2%}  "
          f"equity_risk_premium={MARKET_RISK_PREMIUM_SOURCE:.2%}", file=sys.stderr)
    print(f"Terminal growth (app/valuation/valuation_pipeline.py DEFAULT_TERMINAL_GROWTH_RATE, "
          f"same for every ticker regardless of sector): see per-row value below", file=sys.stderr)

    market_history = _tz_naive(yf.Ticker(MARKET_BENCHMARK).history(period="5y"))

    rows = []
    for i, (ticker, category) in enumerate(TICKERS.items(), 1):
        try:
            row = audit_one(ticker, category, as_of_date, market_history)
        except Exception as e:
            row = {"ticker": ticker, "category": category, "error": str(e)}
        rows.append(row)
        status = row.get("error") or f"iv/price={row.get('iv_to_price')}"
        print(f"[{i}/{len(TICKERS)}] {ticker} -> {status}", file=sys.stderr)

    # ---------------- per-ticker table ----------------
    print()
    header = (f"{'Ticker':<7}{'Category':<32}{'Beta':>6}{'RawWACC':>9}{'WACC':>8}{'Floor?':>7}"
              f"{'TermG':>7}{'CoE':>7}{'IV':>12}{'Price':>10}{'IV/Px':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("error"):
            print(f"{r['ticker']:<7}{r['category']:<32}ERROR: {r['error']}")
            continue
        if not r["dcf_available"]:
            print(f"{r['ticker']:<7}{r['category']:<32}DCF UNAVAILABLE: {r['dcf_unavailable_reason']}")
            continue
        print(
            f"{r['ticker']:<7}{r['category']:<32}{r['beta']:>6.2f}"
            f"{r['raw_wacc']*100:>8.2f}%{r['wacc_used']*100:>7.2f}%{str(r['wacc_floored']):>7}"
            f"{r['terminal_growth_rate']*100:>6.1f}%{r['cost_of_equity']*100:>6.2f}%"
            f"{r['intrinsic_value']:>12,.2f}{r['price_as_of']:>10,.2f}{r['iv_to_price']:>8.2f}"
        )

    # ---------------- CAPM input sources ----------------
    print("\nCAPM INPUT SOURCES:")
    print(f"  Risk-free rate:      {RISK_FREE_RATE_SOURCE:.2%}  -- HARDCODED default in WACCEngine.__init__ "
          f"(app/valuation/wacc_engine.py); not fetched from any live Treasury-yield source; identical for "
          f"every ticker and every as-of date tested.")
    print(f"  Equity risk premium: {MARKET_RISK_PREMIUM_SOURCE:.2%}  -- HARDCODED default in the same "
          f"WACCEngine.__init__ signature; not sourced from any live/historical ERP estimate; identical "
          f"for every ticker and every as-of date tested.")
    print(f"  Beta:                per-ticker, computed by phase2_backtest.py's own _trailing_beta() -- "
          f"OLS-style covariance/variance of trailing daily returns vs. {MARKET_BENCHMARK} ending at the "
          f"as-of date ({250}-trading-day window), falling back to 1.2 if fewer than 60 aligned trading "
          f"days are available. (The live, non-backtest pipeline instead uses yfinance's own reported "
          f"beta when present, same 1.2 fallback otherwise -- see market_data_tool.py.)")
    print(f"  Terminal growth:     app/valuation/valuation_pipeline.py's DEFAULT_TERMINAL_GROWTH_RATE -- "
          f"a single fixed value applied to every ticker regardless of sector/company (see per-row TermG "
          f"column above; that module's own docstring documents this was raised from 3% to 4% after an "
          f"earlier backtest found a systematic downward bias, explicitly flagged there as unverified "
          f"whether 4% alone closes the gap).")

    # ---------------- intrinsic-value-to-price distribution ----------------
    valid = [r for r in rows if not r.get("error") and r.get("dcf_available") and r.get("iv_to_price") is not None]
    print(f"\nINTRINSIC-VALUE-TO-PRICE DISTRIBUTION ({len(valid)} DCF-available tickers "
          f"of {len(rows)} total, as of {as_of_date.date()}):")
    if valid:
        ratios = sorted(r["iv_to_price"] for r in valid)
        mean_r = sum(ratios) / len(ratios)
        median_r = statistics.median(ratios)
        below_1 = sum(1 for x in ratios if x < 1.0)
        print(f"  Mean:   {mean_r:.3f}")
        print(f"  Median: {median_r:.3f}")
        print(f"  Min:    {ratios[0]:.3f} ({[r['ticker'] for r in valid if r['iv_to_price'] == ratios[0]][0]})")
        print(f"  Max:    {ratios[-1]:.3f} ({[r['ticker'] for r in valid if r['iv_to_price'] == ratios[-1]][0]})")
        pcts = [10, 25, 50, 75, 90]
        try:
            import numpy as np
            pct_values = np.percentile(ratios, pcts)
            print("  Percentiles: " + ", ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, pct_values)))
        except ImportError:
            pass
        print(f"  Below 1.0 (DCF says overvalued): {below_1}/{len(ratios)} = {100*below_1/len(ratios):.1f}%")
        print()
        if median_r < 1.0:
            print(f"  MEDIAN IS BELOW 1.0 ({median_r:.3f}) -- across a universe deliberately spanning "
                  f"deep-value, mid-cap, hypergrowth, mega-cap, and financial names, a systematic skew "
                  f"toward the DCF calling companies overvalued (rather than a roughly even split, which "
                  f"is what an unbiased estimator applied to a broad, diverse universe should produce) "
                  f"points at the assumptions (WACC/CAPM inputs, terminal growth), not at any individual "
                  f"company being genuinely overvalued.")
        else:
            print(f"  Median is at or above 1.0 ({median_r:.3f}) -- no evidence in this run of the "
                  f"systematic downward bias documented in valuation_pipeline.py's own module docstring "
                  f"for the pre-4%-terminal-growth version of this same check.")
    else:
        print("  No DCF-available tickers to compute a distribution from.")


if __name__ == "__main__":
    main()
