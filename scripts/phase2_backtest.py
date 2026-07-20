"""
phase2_backtest.py

Historical backtest of the recommendation pipeline. Standalone --
does not touch the Streamlit UI.

No-look-ahead design
---------------------
yfinance has no "give me the database as it looked on date X" API --
it always returns the CURRENT trailing ~4-5 fiscal years of annual
statements and the full price history, regardless of when you ask.
To simulate "run the pipeline as of 12 months ago" without look-ahead,
this script:

  1. Fetches the full current price history and the full current
     financial-statement history (both are ALWAYS fetched fresh --
     the filtering below is what makes the *use* of that data
     point-in-time, not the fetch itself).
  2. Drops any fiscal year whose 10-K would not plausibly have been
     PUBLIC yet as of the as-of date -- a fiscal year ending on date
     E is treated as known only once E + FILING_LAG_DAYS <= as_of_date
     (90 days is a conservative buffer for large-filer 10-K deadlines).
  3. Uses the price ON the as-of date (nearest prior close) for
     everything the valuation depends on -- current_price, market cap
     (price_as_of x shares outstanding from the point-in-time
     financials), and a BETA COMPUTED FROM TRAILING PRICE HISTORY
     ending at the as-of date (not today's yfinance beta, which can
     reflect the last 12 months of trading the backtest is trying to
     predict).
  4. Only pulls TODAY's price once, at the very end, purely to compute
     realized return -- it never feeds back into the valuation.

Scope limitation, disclosed rather than silently cut: sentiment
(SEC-filing FinBERT + news FinBERT) is NOT recomputed point-in-time
here. Doing that properly means resolving which specific SEC filing
was public as of each historical as-of date and running the local LLM
pipeline per ticker -- a much larger, much slower undertaking than
this accuracy check needs, since sentiment only affects the small
subset of tickers where DCF is unavailable (the Insufficient-Data
fallback path). For those tickers, this backtest passes no sentiment,
so the fallback recommendation is relative-valuation-only. This
matches the live pipeline's own graceful-degradation behavior when
sentiment truly is unavailable, it just isn't attempting a historical
sentiment reconstruction.
"""

import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from app.core.research_context import ResearchContext
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.tools.valuation_tool import ValuationTool
from app.reporting.report_data_builder import derive_recommendation, compute_signal_agreement
from app.valuation.ml_features import extract_features

FILING_LAG_DAYS = 90
BACKTEST_MONTHS_AGO = 12
BUY_THRESHOLD = 5.0
SELL_THRESHOLD = -5.0
BETA_WINDOW_TRADING_DAYS = 252
MARKET_BENCHMARK = "^GSPC"

TICKERS = {
    # Deep-value / low-multiple
    "XOM": "deep-value (energy)", "CVX": "deep-value (energy)", "HAL": "deep-value (energy services)",
    "MO": "deep-value (tobacco)", "PM": "deep-value (tobacco)",
    "T": "deep-value (telecom)", "VZ": "deep-value (telecom)",
    "KHC": "deep-value (packaged food)", "INTC": "deep-value/distressed (semis)",
    "F": "deep-value (auto)", "GM": "deep-value (auto)", "WBA": "deep-value (retail pharmacy)",
    "HPQ": "deep-value (hardware)", "UPS": "deep-value (logistics)", "NEM": "deep-value (mining)",

    # Mid-caps outside tech/finance
    "CMI": "mid-cap (industrials)", "DPZ": "mid-cap (consumer disc.)",
    "DAL": "mid-cap (airlines)", "WHR": "mid-cap (consumer durables)",
    "HAS": "mid-cap (consumer disc.)", "LUV": "mid-cap (airlines)",
    "NUE": "mid-cap (steel/industrials)", "CAG": "mid-cap (consumer staples)",
    "CLX": "mid-cap (consumer staples)", "HRL": "mid-cap (consumer staples)",
    "KMB": "mid-cap (consumer staples)", "YUM": "mid-cap (consumer disc.)",
    "PHM": "mid-cap (homebuilder)", "DE": "mid-cap (industrials)",
    "CAT": "mid-cap (industrials)", "EMR": "mid-cap (industrials)",

    # Hypergrowth / negative-FCF / plausibly overvalued
    "PLTR": "hypergrowth/plausibly overvalued", "CVNA": "formerly distressed/high-multiple",
    "SMCI": "distressed (accounting issues)", "RIVN": "hypergrowth/negative-FCF (EV)",
    "LCID": "hypergrowth/negative-FCF (EV)", "SNOW": "hypergrowth/high-multiple",
    "U": "hypergrowth/negative-FCF", "COIN": "high-volatility/high-multiple",
    "MSTR": "high-multiple/distressed-adjacent", "AI": "hypergrowth/negative-FCF",
    "UPST": "hypergrowth/negative-FCF (fintech)", "AFRM": "hypergrowth/negative-FCF (fintech)",
    "RBLX": "hypergrowth/negative-FCF (gaming)", "NIO": "hypergrowth/negative-FCF (EV)",
    "DKNG": "hypergrowth/negative-FCF (gaming)", "PATH": "hypergrowth/high-multiple",

    # Mega-cap
    "AAPL": "mega-cap", "NVDA": "mega-cap", "MSFT": "mega-cap",
    "GOOGL": "mega-cap", "BLK": "mega-cap", "META": "mega-cap",
    "AMZN": "mega-cap", "TSLA": "mega-cap (high-multiple)",
    "ORCL": "mega-cap", "CRM": "mega-cap", "ADBE": "mega-cap", "NFLX": "mega-cap",
    "V": "mega-cap (payments)", "MA": "mega-cap (payments)",

    # Financial sector -- banks/insurers specifically, to stress-test
    # the "Insufficient Data" fallback path at scale (only JPM had hit
    # it before this expansion, since financials structurally don't
    # report EBIT/capex/current-vs-non-current assets the way
    # non-financial companies do -- see fcff_engine.py).
    "JPM": "financial (bank)", "BAC": "financial (bank)", "WFC": "financial (bank)",
    "C": "financial (bank)", "GS": "financial (bank)", "MS": "financial (bank)",
    "USB": "financial (bank)", "PNC": "financial (bank)", "TFC": "financial (bank)",
    "SCHW": "financial (brokerage)", "AXP": "financial (payments/credit)",
    "AIG": "financial (insurer)", "MET": "financial (insurer)", "PRU": "financial (insurer)",
    "TRV": "financial (insurer)", "ALL": "financial (insurer)", "PGR": "financial (insurer)",
    "CB": "financial (insurer)",
}


def _tz_naive(df):
    if df is None or df.empty:
        return df
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df


def _price_on_or_before(price_history, date):
    prices = price_history["Close"]
    prices = prices[prices.index <= date]
    if prices.empty:
        return None
    return float(prices.iloc[-1])


def _trailing_beta(stock_history, market_history, as_of_date):
    stock_returns = stock_history["Close"].pct_change()
    market_returns = market_history["Close"].pct_change()
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned.columns = ["stock", "market"]
    aligned = aligned[aligned.index <= as_of_date].tail(BETA_WINDOW_TRADING_DAYS)
    if len(aligned) < 60:
        return None
    variance = aligned["market"].var()
    if not variance:
        return None
    return aligned["stock"].cov(aligned["market"]) / variance


def _point_in_time_statement(raw_df, as_of_date):
    keep_cols = [
        c for c in raw_df.columns
        if (c + pd.Timedelta(days=FILING_LAG_DAYS)) <= pd.Timestamp(as_of_date)
    ]
    return raw_df[keep_cols]


def run_one(ticker, category, as_of_date, today_date, market_history):
    stock = yf.Ticker(ticker)

    # 10y, not 2y: RelativeValuationEngine looks up a year-end close
    # for each fiscal year in the point-in-time financials (up to ~4
    # years back from the as-of date), not just the as-of/today pair
    # -- a shorter window silently starves it of data and it returns
    # None for every ticker (caught in an earlier run of this script).
    price_history = _tz_naive(stock.history(period="10y"))
    if price_history is None or price_history.empty:
        raise ValueError("no price history available")

    price_as_of = _price_on_or_before(price_history, as_of_date)
    price_today = _price_on_or_before(price_history, today_date)
    if price_as_of is None or price_today is None:
        raise ValueError("insufficient price history spanning as-of date and today")

    realized_return_pct = (price_today - price_as_of) / price_as_of * 100

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
        raise ValueError("no fiscal year was filed early enough to be known as of the as-of date")

    financial_df = FinancialStatementNormaliser(income_pit, balance_pit, cashflow_pit).normalise()
    if financial_df.empty or len(financial_df) < 2:
        raise ValueError("insufficient point-in-time financial history (need >=2 fiscal years)")

    shares_outstanding = None
    if "shares_outstanding" in financial_df.columns:
        series = financial_df["shares_outstanding"].dropna()
        if not series.empty:
            shares_outstanding = series.iloc[-1]
    if not shares_outstanding:
        raise ValueError("no point-in-time shares outstanding")

    market_cap_as_of = price_as_of * shares_outstanding

    ctx = ResearchContext(
        ticker=ticker,
        question=f"Should I buy {ticker}? (backtest as-of {as_of_date.date()})",
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

    # Sentiment intentionally omitted -- see module docstring scope note.
    rec = derive_recommendation(ctx.valuation_results, None, None)
    relative = ctx.valuation_results.get("relative_valuation")

    # Since derive_recommendation now downgrades an actual Buy/Sell
    # disagreement to Hold outright (not just a confidence flag), a
    # SCORED Buy/Sell row can no longer be a "disagree" case -- that
    # combination is structurally impossible now. What's newly
    # measurable instead: for rows where DCF's own call WOULD have
    # disagreed with relative valuation (dcf_only_rating), was
    # forcing it to Hold the right call, or would the original
    # DCF-only direction have scored better?
    agreement = compute_signal_agreement(rec.get("dcf_only_rating"), relative)

    # ML feature vector at this same point-in-time context, for
    # scripts/build_ml_training_set.py -- reuses this function's
    # already-validated point-in-time ctx instead of reconstructing
    # it a second time. None where DCF was unavailable (see
    # extract_features' own docstring).
    ml_features = extract_features(ctx)

    return {
        "ticker": ticker,
        "category": category,
        "as_of_date": as_of_date.date().isoformat(),
        "recommendation": rec["rating"],
        "dcf_only_rating": rec.get("dcf_only_rating"),
        "downgraded": rec.get("downgraded_for_disagreement", False),
        "dcf_available": ctx.valuation_results.get("dcf_available"),
        "upside_pct": ctx.valuation_results.get("upside_percent"),
        "relative_signal": relative["signal"] if relative else None,
        "agreement": agreement,
        "price_as_of": round(price_as_of, 2),
        "price_today": round(price_today, 2),
        "realized_return_pct": round(realized_return_pct, 2),
        "ml_features": ml_features,
        "error": None,
    }


def score_rating(rating, realized_return_pct):
    """Scores an arbitrary rating string against a realized return --
    factored out of score() so the same accuracy definition can also
    be applied to a hypothetical rating (e.g. what dcf_only_rating
    would have scored, had it not been downgraded)."""
    if rating is None or rating == "Insufficient Data" or realized_return_pct is None:
        return None
    if rating == "Buy":
        return realized_return_pct > BUY_THRESHOLD
    if rating == "Sell":
        return realized_return_pct < SELL_THRESHOLD
    if rating == "Hold":
        return SELL_THRESHOLD <= realized_return_pct <= BUY_THRESHOLD
    return None


def score(row):
    """Returns True/False (correct/incorrect), or None if not scored
    (Insufficient Data, or the run itself errored)."""
    if row.get("error"):
        return None
    return score_rating(row["recommendation"], row["realized_return_pct"])


def main():
    today_date = pd.Timestamp(datetime.utcnow().date())
    as_of_date = today_date - pd.Timedelta(days=BACKTEST_MONTHS_AGO * 30)

    print(f"As-of date: {as_of_date.date()}   Today: {today_date.date()}", file=sys.stderr)
    print(f"Accuracy bands: Buy>+{BUY_THRESHOLD}%  Sell<{SELL_THRESHOLD}%  "
          f"Hold in [{SELL_THRESHOLD}%,+{BUY_THRESHOLD}%]", file=sys.stderr)

    market_history = _tz_naive(yf.Ticker(MARKET_BENCHMARK).history(period="2y"))

    rows = []
    for ticker, category in TICKERS.items():
        print(f"Running {ticker} ({category})...", file=sys.stderr)
        try:
            row = run_one(ticker, category, as_of_date, today_date, market_history)
        except Exception as e:
            row = {
                "ticker": ticker, "category": category, "as_of_date": as_of_date.date().isoformat(),
                "recommendation": None, "dcf_only_rating": None, "downgraded": False,
                "dcf_available": None, "upside_pct": None,
                "relative_signal": None, "agreement": None, "price_as_of": None,
                "price_today": None, "realized_return_pct": None, "error": str(e),
            }
            traceback.print_exc(file=sys.stderr)
        rows.append(row)

    for row in rows:
        row["correct"] = score(row)

    # ---------------- results table ----------------
    print()
    header = f"{'Ticker':<7}{'Category':<32}{'Rec':<9}{'Downgr':<7}{'DCF?':<6}{'Upside%':>9}  {'RelSig':<10}{'AsOf$':>9}{'Today$':>9}{'Return%':>9}  Correct"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("error"):
            print(f"{r['ticker']:<7}{r['category']:<32}ERROR: {r['error']}")
            continue
        upside = f"{r['upside_pct']:.1f}" if isinstance(r["upside_pct"], (int, float)) else "N/A"
        correct = "N/A" if r["correct"] is None else ("YES" if r["correct"] else "NO")
        print(f"{r['ticker']:<7}{r['category']:<32}{str(r['recommendation']):<9}"
              f"{str(r['downgraded']):<7}{str(r['dcf_available']):<6}{upside:>9}  "
              f"{str(r['relative_signal']):<10}{r['price_as_of']:>9}{r['price_today']:>9}"
              f"{r['realized_return_pct']:>9}  {correct}")

    # ---------------- summary ----------------
    valid = [r for r in rows if not r.get("error")]
    scored = [r for r in valid if r["correct"] is not None]
    insufficient = [r for r in valid if r["recommendation"] == "Insufficient Data"]
    errored = [r for r in rows if r.get("error")]

    print()
    print(f"Total tickers: {len(rows)}   Ran clean: {len(valid)}   Errored: {len(errored)}   "
          f"Scored: {len(scored)}   Insufficient Data (unscored): {len(insufficient)}")

    if scored:
        overall_acc = 100 * sum(1 for r in scored if r["correct"]) / len(scored)
        print(f"\nOVERALL ACCURACY: {overall_acc:.1f}% ({sum(1 for r in scored if r['correct'])}/{len(scored)})")

        print("\nBy recommendation type:")
        for rec_type in ("Buy", "Hold", "Sell"):
            group = [r for r in scored if r["recommendation"] == rec_type]
            if group:
                acc = 100 * sum(1 for r in group if r["correct"]) / len(group)
                print(f"  {rec_type:<6} {acc:5.1f}% ({sum(1 for r in group if r['correct'])}/{len(group)})")

        print("\nBy DCF availability:")
        for avail, label in ((True, "DCF-based calls"), (False, "Fallback (relative-valuation) calls")):
            group = [r for r in scored if r["dcf_available"] == avail]
            if group:
                acc = 100 * sum(1 for r in group if r["correct"]) / len(group)
                print(f"  {label:<38} {acc:5.1f}% ({sum(1 for r in group if r['correct'])}/{len(group)})")

        # Disagreement now forces Hold outright (see report_data_builder.py),
        # so a scored Buy/Sell can no longer be a "disagree" case -- that
        # combination is structurally impossible. What IS newly testable:
        # for every row where DCF's own call would have disagreed with
        # relative valuation, was downgrading to Hold the right move, or
        # would trusting the original DCF-only direction have scored better?
        downgraded_rows = [r for r in valid if r.get("downgraded")]
        if downgraded_rows:
            print(f"\nDowngrade guardrail ({len(downgraded_rows)} tickers had DCF/relative disagreement "
                  f"and were downgraded to Hold) -- did downgrading help?")
            hold_scores = [score_rating("Hold", r["realized_return_pct"]) for r in downgraded_rows]
            original_scores = [score_rating(r["dcf_only_rating"], r["realized_return_pct"]) for r in downgraded_rows]
            hold_valid = [s for s in hold_scores if s is not None]
            original_valid = [s for s in original_scores if s is not None]
            if hold_valid:
                acc = 100 * sum(hold_valid) / len(hold_valid)
                print(f"  As downgraded Hold:              {acc:5.1f}% ({sum(hold_valid)}/{len(hold_valid)})")
            if original_valid:
                acc = 100 * sum(original_valid) / len(original_valid)
                print(f"  As original DCF-only call:       {acc:5.1f}% ({sum(original_valid)}/{len(original_valid)})")

    if insufficient:
        print(f"\nInsufficient Data cases ({len(insufficient)}) -- not scored against the accuracy definition, "
              f"reported separately since the recommendation intentionally deferred:")
        for r in insufficient:
            print(f"  {r['ticker']:<7} realized_return={r['realized_return_pct']:+.1f}%  "
                  f"relative_signal={r['relative_signal']}")

    if errored:
        print(f"\nErrored tickers ({len(errored)}):")
        for r in errored:
            print(f"  {r['ticker']:<7} {r['error']}")

    # ---------------- market base-rate comparison ----------------
    # Same ticker universe, same 12-month window -- what fraction
    # would have qualified as "up" / "down" / "flat" by the same
    # +/-5% bands, independent of what the tool recommended? This is
    # the naive baseline the tool's Buy/Sell accuracy has to beat to
    # mean anything more than "the market went up/down anyway."
    returns = [r["realized_return_pct"] for r in valid if r["realized_return_pct"] is not None]
    if returns:
        up = sum(1 for x in returns if x > BUY_THRESHOLD)
        down = sum(1 for x in returns if x < SELL_THRESHOLD)
        flat = len(returns) - up - down
        print(f"\nMARKET BASE RATE (same {len(returns)} tickers, same 12mo window, independent of any recommendation):")
        print(f"  Up   (>+{BUY_THRESHOLD:.0f}%):  {100*up/len(returns):5.1f}% ({up}/{len(returns)})")
        print(f"  Down (<{SELL_THRESHOLD:.0f}%):  {100*down/len(returns):5.1f}% ({down}/{len(returns)})")
        print(f"  Flat ({SELL_THRESHOLD:.0f}% to +{BUY_THRESHOLD:.0f}%): {100*flat/len(returns):5.1f}% ({flat}/{len(returns)})")
        print(f"  (A tool with no real skill that always said \"Buy\" would score ~{100*up/len(returns):.1f}% on this "
              f"universe/window; always \"Sell\" would score ~{100*down/len(returns):.1f}%.)")


if __name__ == "__main__":
    main()
