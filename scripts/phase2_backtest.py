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

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf

from app.analysis.baseline_scoring import BUY_THRESHOLD, SELL_THRESHOLD, naive_baseline_accuracy, score_rating
from app.core.research_context import ResearchContext
from app.data.financial_normalizer import FinancialStatementNormaliser
from app.tools.valuation_tool import ValuationTool
from app.reporting.report_data_builder import derive_recommendation, compute_signal_agreement
from app.valuation.ml_features import extract_features

FILING_LAG_DAYS = 90
BACKTEST_MONTHS_AGO = 12
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

    # derive_recommendation no longer forces a downgrade to Hold on
    # DCF/relative-valuation disagreement (removed after this same
    # backtest showed the forced Hold scored 15.4% vs. 53.8% for
    # trusting DCF's own call -- see report_data_builder.py). A SCORED
    # Buy/Sell row CAN be a "disagree" case again now.
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
        "signal_disagreement": rec.get("signal_disagreement", False),
        "dcf_available": ctx.valuation_results.get("dcf_available"),
        "upside_pct": ctx.valuation_results.get("upside_percent"),
        "relative_signal": relative["signal"] if relative else None,
        "agreement": agreement,
        # Raw component scores (see report_data_builder.py's _dcf_score/
        # _relative_score/_composite_score) -- captured so threshold
        # and weighting experiments (analyze different BUY_THRESHOLD/
        # SELL_THRESHOLD/DCF_WEIGHT/RELATIVE_WEIGHT combinations) can be
        # run by recomputing composite_score = w1*dcf_score +
        # w2*relative_score from this same backtest data, instead of
        # re-fetching and re-running the pipeline for every combination
        # tried.
        "dcf_score": rec.get("dcf_score"),
        "relative_score": rec.get("relative_score"),
        "composite_score": rec.get("composite_score"),
        "price_as_of": round(price_as_of, 2),
        "price_today": round(price_today, 2),
        "realized_return_pct": round(realized_return_pct, 2),
        "ml_features": ml_features,
        "error": None,
    }


def score(row):
    """Returns True/False (correct/incorrect), or None if not scored
    (Insufficient Data, or the run itself errored)."""
    if row.get("error"):
        return None
    return score_rating(row["recommendation"], row["realized_return_pct"])


# ---------------------------------------------------------- baselines & precision/recall
#
# All of this operates on `scored` rows (see main()) -- the exact same
# set OVERALL ACCURACY is computed on, so "the model beat/lost to
# always-Buy by N points" is a real apples-to-apples comparison, not
# two different denominators that happen to look similar. score_rating's
# three branches (Buy: >BUY_THRESHOLD, Sell: <SELL_THRESHOLD, Hold: the
# closed band between) are mutually exclusive and jointly exhaustive
# over any real-valued realized_return_pct -- every function below
# leans on that.

def _true_class(realized_return_pct):
    """The actual outcome a prediction is trying to hit, using the
    exact same bands score_rating() already scores Buy/Hold/Sell
    against -- Buy is "trying to predict Up", Hold "Flat", Sell "Down".
    Not a new definition invented for this: it's score_rating's own
    three branches, named and exposed so precision/recall can be
    computed against them directly instead of only ever collapsing to
    a single correct/incorrect bit."""
    if realized_return_pct > BUY_THRESHOLD:
        return "Up"
    if realized_return_pct < SELL_THRESHOLD:
        return "Down"
    return "Flat"


_PREDICTED_TO_ACTUAL = {"Buy": "Up", "Hold": "Flat", "Sell": "Down"}


def _base_rate(scored_rows):
    """{"Up": n, "Down": n, "Flat": n} -- what fraction of `scored_rows`
    actually landed in each band, independent of what the model said.
    Always-Buy/Always-Hold/Always-Sell's accuracy (below) is this exact
    same count, restated as a score against the model's own scoring
    rule -- computing both from one pass keeps them from silently
    drifting apart."""
    counts = {"Up": 0, "Down": 0, "Flat": 0}
    for r in scored_rows:
        counts[_true_class(r["realized_return_pct"])] += 1
    return counts


def _random_uniform_baseline_accuracy(scored_rows, trials=2000, seed=42):
    """
    Monte Carlo estimate (fixed seed -- reproducible across runs, not
    a different number every time this script executes) of a uniform-
    random Buy/Hold/Sell guesser's accuracy on this exact sample.

    Worth stating plainly: score_rating's three bands are mutually
    exclusive and exhaustive, so a uniform guess's TRUE expected
    accuracy is exactly 1/3 (33.3%) on ANY sample, regardless of its
    actual up/flat/down mix -- this isn't estimating something unknown,
    it's demonstrating that a naive "spread your guesses evenly" rater
    carries no information about the specific tickers under test.
    Simulating it anyway (many trials, averaged) gives a concrete,
    reproducible number for this run instead of only asserting the
    theory, and its closeness to 33.3% is itself a check that the
    simulation is doing what it claims.
    """
    rng = random.Random(seed)
    trial_accuracies = []
    for _ in range(trials):
        correct = 0
        n = 0
        for r in scored_rows:
            guess = rng.choice(("Buy", "Hold", "Sell"))
            result = score_rating(guess, r["realized_return_pct"])
            if result is None:
                continue
            n += 1
            correct += result
        if n:
            trial_accuracies.append(100 * correct / n)
    if not trial_accuracies:
        return None
    return sum(trial_accuracies) / len(trial_accuracies)


def _precision_recall(scored_rows):
    """
    Per-class precision/recall for the Buy/Hold/Sell predictions
    against the Up/Flat/Down actual outcome (_PREDICTED_TO_ACTUAL) --
    a real 3x3 confusion matrix instead of collapsing everything into
    one accuracy number, so (for example) a model that's precise on
    Buy but never actually catches most real Up moves (low recall)
    doesn't look identical to one that's both precise and complete.

    precision = correct / times the model predicted this class
    recall    = correct / times this class actually happened
    Either can be None (not "0%") when its own denominator is zero --
    a class the model never predicted has undefined precision, not 0%.
    """
    results = {}
    for pred_class, actual_class in _PREDICTED_TO_ACTUAL.items():
        predicted_rows = [r for r in scored_rows if r["recommendation"] == pred_class]
        actual_rows = [r for r in scored_rows if _true_class(r["realized_return_pct"]) == actual_class]
        true_positives = sum(
            1 for r in predicted_rows if _true_class(r["realized_return_pct"]) == actual_class
        )
        results[pred_class] = {
            "precision": true_positives / len(predicted_rows) if predicted_rows else None,
            "recall": true_positives / len(actual_rows) if actual_rows else None,
            "predicted_n": len(predicted_rows),
            "actual_n": len(actual_rows),
        }
    return results


def _avg_return(rows):
    values = [r["realized_return_pct"] for r in rows if r["realized_return_pct"] is not None]
    return sum(values) / len(values) if values else None


def _return_spread(scored_rows):
    """
    Average realized return of Buy calls minus average of Sell calls --
    a ranking/discrimination metric, not a directional one. Buy/Sell
    accuracy against a fixed +/-5% threshold rewards being loudly
    right and can make a model that's directionally timid (or just
    unlucky on threshold placement) look like it has no skill, even if
    it reliably ranks Buys above Sells. A model that's mediocre on the
    binary labels but shows a clearly positive spread here IS carrying
    real information -- it just isn't being rewarded for it by a hard
    threshold. Returns (buy_avg, sell_avg, spread), any of which can be
    None if that class was never predicted.
    """
    buy_avg = _avg_return([r for r in scored_rows if r["recommendation"] == "Buy"])
    sell_avg = _avg_return([r for r in scored_rows if r["recommendation"] == "Sell"])
    spread = (buy_avg - sell_avg) if buy_avg is not None and sell_avg is not None else None
    return buy_avg, sell_avg, spread


def _parse_args():
    parser = argparse.ArgumentParser(description="Point-in-time backtest of the recommendation pipeline.")
    # AS_OF_MONTHS_AGO / EXIT_MONTHS_AGO let this same point-in-time
    # methodology test a genuinely independent, non-overlapping
    # historical window instead of always "12 months ago to today" --
    # e.g. `phase2_backtest.py 24 12` prices/values companies as of 24
    # months ago and measures realized return through 12 months ago
    # (still a 12-month holding period, just anchored to a different
    # point in market history, so it isn't confounded with
    # holding-period length the way "just change months-ago to 24"
    # would be). Defaults preserve the original behavior exactly.
    parser.add_argument("as_of_months_ago", nargs="?", type=int, default=BACKTEST_MONTHS_AGO)
    parser.add_argument("exit_months_ago", nargs="?", type=int, default=0)
    parser.add_argument(
        "--universe", type=str, default=None,
        help="Path to a {ticker: category} JSON file (e.g. scripts/ticker_universe.json) "
             "to run against instead of the ~80-ticker hand-curated TICKERS dict.",
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Concurrent worker threads (default: %(default)s). This is network-bound "
             "(yfinance calls per ticker), not CPU-bound, so threads are the right tool -- "
             "kept moderate to avoid Yahoo Finance rate-limiting on a large universe.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    as_of_months_ago = args.as_of_months_ago
    exit_months_ago = args.exit_months_ago

    if args.universe:
        with open(args.universe) as f:
            tickers = json.load(f)
        universe_tag = Path(args.universe).stem
    else:
        tickers = TICKERS
        universe_tag = "curated"

    today_date = pd.Timestamp(datetime.utcnow().date())
    as_of_date = today_date - pd.Timedelta(days=as_of_months_ago * 30)
    exit_date = today_date - pd.Timedelta(days=exit_months_ago * 30)

    results_output_path = str(
        Path(__file__).resolve().parent
        / f"backtest_results_{universe_tag}_asof{as_of_months_ago}mo_exit{exit_months_ago}mo.json"
    )

    print(f"Universe: {universe_tag} ({len(tickers)} tickers)   Workers: {args.workers}", file=sys.stderr)
    print(f"As-of date: {as_of_date.date()}   Exit date: {exit_date.date()}", file=sys.stderr)
    print(f"Accuracy bands: Buy>+{BUY_THRESHOLD}%  Sell<{SELL_THRESHOLD}%  "
          f"Hold in [{SELL_THRESHOLD}%,+{BUY_THRESHOLD}%]", file=sys.stderr)

    # 5y, not 2y: a trailing beta window (BETA_WINDOW_TRADING_DAYS)
    # computed AS OF an as_of_date that's itself far in the past (e.g.
    # 24 months ago) needs benchmark price history reaching back
    # further still -- 2y was only ever enough for the original
    # "12 months ago" default.
    market_history = _tz_naive(yf.Ticker(MARKET_BENCHMARK).history(period="5y"))

    def _run(ticker, category):
        try:
            return run_one(ticker, category, as_of_date, exit_date, market_history)
        except Exception as e:
            return {
                "ticker": ticker, "category": category, "as_of_date": as_of_date.date().isoformat(),
                "recommendation": None, "dcf_only_rating": None, "signal_disagreement": False,
                "dcf_available": None, "upside_pct": None,
                "relative_signal": None, "agreement": None,
                "dcf_score": None, "relative_score": None, "composite_score": None,
                "price_as_of": None,
                "price_today": None, "realized_return_pct": None, "error": str(e),
            }

    rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run, ticker, category): (ticker, category) for ticker, category in tickers.items()}
        for future in as_completed(futures):
            ticker, category = futures[future]
            row = future.result()
            rows.append(row)
            completed += 1
            status = row.get("error") or row.get("recommendation")
            print(f"[{completed}/{len(tickers)}] {ticker} ({category}) -> {status}", file=sys.stderr)

    for row in rows:
        row["correct"] = score(row)

    # ml_features is a large per-row dict only build_ml_training_set.py
    # needs -- excluded here to keep this export focused on what the
    # threshold/weighting grid search actually uses.
    with open(results_output_path, "w") as f:
        json.dump([{k: v for k, v in row.items() if k != "ml_features"} for row in rows], f, indent=2)
    print(f"\nRaw results saved -> {results_output_path}", file=sys.stderr)

    # ---------------- results table ----------------
    print()
    header = f"{'Ticker':<7}{'Category':<32}{'Rec':<9}{'Disagr':<7}{'DCF?':<6}{'Upside%':>9}  {'RelSig':<10}{'AsOf$':>9}{'Today$':>9}{'Return%':>9}  Correct"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("error"):
            print(f"{r['ticker']:<7}{r['category']:<32}ERROR: {r['error']}")
            continue
        upside = f"{r['upside_pct']:.1f}" if isinstance(r["upside_pct"], (int, float)) else "N/A"
        correct = "N/A" if r["correct"] is None else ("YES" if r["correct"] else "NO")
        print(f"{r['ticker']:<7}{r['category']:<32}{str(r['recommendation']):<9}"
              f"{str(r['signal_disagreement']):<7}{str(r['dcf_available']):<6}{upside:>9}  "
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

        # ---- naive baselines + base rate (same `scored` set/denominator as above) ----
        base_rate = _base_rate(scored)
        n_scored = len(scored)
        print(f"\nBASE RATE ({n_scored} scored tickers, independent of any recommendation):")
        print(f"  Up   (>+{BUY_THRESHOLD:.0f}%): {100*base_rate['Up']/n_scored:5.1f}% ({base_rate['Up']}/{n_scored})")
        print(f"  Down (<{SELL_THRESHOLD:.0f}%): {100*base_rate['Down']/n_scored:5.1f}% ({base_rate['Down']}/{n_scored})")
        print(f"  Flat ({SELL_THRESHOLD:.0f}% to +{BUY_THRESHOLD:.0f}%): {100*base_rate['Flat']/n_scored:5.1f}% ({base_rate['Flat']}/{n_scored})")

        print(f"\nNAIVE BASELINES (scored the same way as OVERALL ACCURACY, same {n_scored} tickers):")
        fixed_baseline_accs = {}
        for label, fixed_rating in (("Always Buy", "Buy"), ("Always Hold", "Hold"), ("Always Sell", "Sell")):
            acc = naive_baseline_accuracy(scored, fixed_rating)
            fixed_baseline_accs[fixed_rating] = acc
            print(f"  {label:<18} {acc:5.1f}%")
        random_acc = _random_uniform_baseline_accuracy(scored)
        print(f"  {'Random (uniform)':<18} {random_acc:5.1f}%  (2000-trial Monte Carlo, seed=42 -- "
              f"theoretical expectation is exactly 33.3% on any sample, since Buy/Hold/Sell's bands "
              f"are mutually exclusive and exhaustive)")
        print(f"  (Always-Buy/Hold/Sell above are exactly the BASE RATE percentages, restated as "
              f"accuracy against the model's own scoring rule -- this is what the model has to beat "
              f"to mean anything more than \"the market did X anyway.\")")
        best_fixed_baseline = max(v for v in fixed_baseline_accs.values() if v is not None)
        print(f"  Model vs. best naive baseline: {overall_acc - best_fixed_baseline:+.1f} points")

        # ---- return-magnitude / ranking metric ----
        # See _return_spread's docstring: a ranking signal, not a
        # directional one -- distinct from (and not reducible to)
        # accuracy against the +/-5% threshold above.
        buy_avg, sell_avg, spread = _return_spread(scored)
        print(f"\nRETURN SPREAD (ranking signal, independent of the +/-{BUY_THRESHOLD:.0f}% accuracy threshold):")
        print(f"  Avg realized return of Buy calls:  {buy_avg:+6.2f}%" if buy_avg is not None else "  Avg realized return of Buy calls:  N/A (no Buy calls)")
        print(f"  Avg realized return of Sell calls: {sell_avg:+6.2f}%" if sell_avg is not None else "  Avg realized return of Sell calls: N/A (no Sell calls)")
        if spread is not None:
            print(f"  Spread (Buy avg - Sell avg):       {spread:+6.2f} points "
                  f"({'Buys outperformed Sells -- the ranking carries information even where the binary label does not' if spread > 0 else 'Buys did NOT outperform Sells -- no evidence of ranking skill on this sample'})")
        else:
            print("  Spread: N/A (need both a Buy and a Sell call to compare)")

        # ---- per-class precision / recall ----
        print("\nPER-CLASS PRECISION / RECALL (Buy->Up, Hold->Flat, Sell->Down):")
        print(f"  {'Class':<8}{'Precision':>12}{'Recall':>10}{'Predicted N':>14}{'Actual N':>11}")
        precision_recall = _precision_recall(scored)
        for cls in ("Buy", "Hold", "Sell"):
            stats = precision_recall[cls]
            prec_str = f"{100*stats['precision']:.1f}%" if stats["precision"] is not None else "N/A"
            recall_str = f"{100*stats['recall']:.1f}%" if stats["recall"] is not None else "N/A"
            print(f"  {cls:<8}{prec_str:>12}{recall_str:>10}{stats['predicted_n']:>14}{stats['actual_n']:>11}")

        print("\nBy DCF availability:")
        for avail, label in ((True, "DCF-based calls"), (False, "Fallback (relative-valuation) calls")):
            group = [r for r in scored if r["dcf_available"] == avail]
            if group:
                acc = 100 * sum(1 for r in group if r["correct"]) / len(group)
                print(f"  {label:<38} {acc:5.1f}% ({sum(1 for r in group if r['correct'])}/{len(group)})")

        # DCF/relative disagreement no longer forces a Hold override
        # (removed -- see report_data_builder.py's derive_recommendation
        # docstring for why). What's testable now: on the rows where
        # DCF's own directional call disagreed with relative valuation,
        # does the blended composite rating (current behavior) actually
        # do better than either signal trusted alone?
        disagreement_rows = [r for r in valid if r.get("signal_disagreement")]
        if disagreement_rows:
            print(f"\nSignal disagreement ({len(disagreement_rows)} tickers had DCF/relative "
                  f"disagreement) -- how did each approach score on exactly these rows?")
            blended_scores = [r["correct"] for r in disagreement_rows]
            dcf_only_scores = [score_rating(r["dcf_only_rating"], r["realized_return_pct"]) for r in disagreement_rows]
            hold_scores = [score_rating("Hold", r["realized_return_pct"]) for r in disagreement_rows]
            blended_valid = [s for s in blended_scores if s is not None]
            dcf_only_valid = [s for s in dcf_only_scores if s is not None]
            hold_valid = [s for s in hold_scores if s is not None]
            if blended_valid:
                acc = 100 * sum(blended_valid) / len(blended_valid)
                print(f"  Blended composite rating (current):  {acc:5.1f}% ({sum(blended_valid)}/{len(blended_valid)})")
            if dcf_only_valid:
                acc = 100 * sum(dcf_only_valid) / len(dcf_only_valid)
                print(f"  DCF-only call (ignoring relative):   {acc:5.1f}% ({sum(dcf_only_valid)}/{len(dcf_only_valid)})")
            if hold_valid:
                acc = 100 * sum(hold_valid) / len(hold_valid)
                print(f"  Forced Hold (old guardrail):         {acc:5.1f}% ({sum(hold_valid)}/{len(hold_valid)})")

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


if __name__ == "__main__":
    main()
