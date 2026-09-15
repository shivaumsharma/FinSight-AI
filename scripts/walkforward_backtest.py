"""
walkforward_backtest.py

A real walk-forward PORTFOLIO backtest, not another per-ticker
directional-accuracy check -- see scripts/canonical_accuracy.py for
that. This one builds an actual equity curve: at each of 13 quarterly
rebalance dates spanning the last 3 years, ranks every ticker in a
sector-stratified ~275-ticker sample by the SAME point-in-time
composite_score the live app computes, holds the top TOP_N_HOLDINGS
Buy-rated names equal-weighted until the next rebalance, and charges a
transaction cost on every trade -- then reports CAGR, annualized
volatility, Sharpe, Sortino, max drawdown, hit rate, and profit factor
against two comparators: S&P 500 buy-and-hold, and a naive factor
baseline (equal-weight the whole sampled universe every quarter, no
valuation signal at all -- this backtest's answer to "always-Buy").

Reuses phase2_backtest.py's entire point-in-time discipline (filing-lag
filter, trailing beta, point_in_time_cutoff, risk-free-rate override)
via its _fetch_raw_ticker_data/_score_ticker_at_date split (see that
module's own comment on why it's split that way) -- this script is
exactly the caller that split existed for. Each ticker is fetched
ONCE (not once per rebalance date), then scored 13 times against the
one fetch -- a walk-forward run over N tickers costs the same number
of network calls as a single-snapshot run over N tickers, not 13x
that, which is what made every past attempt to scale this project's
backtests past a few hundred tickers hit real Yahoo Finance
rate-limiting (see this project's own EVALUATION.md and
generate_embedding_training_data.py's docstring).

Scope, stated plainly: the composite formula's own weights/thresholds
(DCF_WEIGHT/RELATIVE_WEIGHT/BUY_THRESHOLD/SELL_THRESHOLD in
report_data_builder.py) are NOT re-tuned per rolling window here --
this tests a frozen strategy's performance rolled forward through
real time with real costs, not a "retrain the model each fold"
walk-forward. That's a real, useful next step, not this one.

Run: python scripts/walkforward_backtest.py [--universe-size 275]
     [--workers 10]
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from app.analysis import portfolio_metrics as pm
from scripts.phase2_backtest import (
    MARKET_BENCHMARK,
    _fetch_raw_ticker_data,
    _score_ticker_at_date,
    _tz_naive,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TICKER_UNIVERSE_PATH = SCRIPT_DIR / "ticker_universe.json"

YEARS_BACK = 3
REBALANCE_MONTHS = 3
N_REBALANCES = (YEARS_BACK * 12) // REBALANCE_MONTHS + 1  # 13 dates, 12 holding periods
TOP_N_HOLDINGS = 25
UNIVERSE_SAMPLE_SEED = 42

STARTING_CAPITAL = 1_000_000.0

# Combined commission + spread + slippage, applied to the FULL
# round-trip value of whatever fraction of the portfolio turns over at
# a rebalance (see app/analysis/portfolio_metrics.turnover's own
# "buy-side-only, 1.0 = full replacement" convention) -- a single,
# transparent, documented modeling assumption, not calibrated against
# real historical bid/ask data (this project has none to calibrate
# against). Configurable here, not buried inline, so it's easy to spot
# and to sensitivity-test later.
TRANSACTION_COST_BPS = 10.0


def _sample_universe(target_size: int, seed: int) -> dict:
    """Sector-stratified sample of `target_size` tickers from the full
    1,002-ticker universe -- same round-robin-across-sectors, fixed-seed
    pattern already used in generate_embedding_training_data.py's
    _build_train_tickers() (reimplemented here, not imported, since
    that module's other concerns -- LLM question generation -- are
    unrelated to this script). Ensures the sample spans every sector
    roughly evenly rather than whichever happen to sort first."""
    with open(TICKER_UNIVERSE_PATH, encoding="utf-8") as f:
        universe = json.load(f)

    by_sector = {}
    for ticker, category_label in universe.items():
        sector = category_label.split(" (")[0]
        by_sector.setdefault(sector, []).append(ticker)

    rng = random.Random(seed)
    for tickers in by_sector.values():
        rng.shuffle(tickers)

    sectors = sorted(by_sector.keys())
    sampled = []
    idx = 0
    while len(sampled) < target_size and any(by_sector[s] for s in sectors):
        sector = sectors[idx % len(sectors)]
        if by_sector[sector]:
            sampled.append(by_sector[sector].pop())
        idx += 1

    return {ticker: universe[ticker] for ticker in sampled}


def _rebalance_dates(today_date: pd.Timestamp) -> list:
    """N_REBALANCES dates, oldest first, ~REBALANCE_MONTHS apart,
    spanning YEARS_BACK -- same days-per-month convention
    (months * 30) phase2_backtest.py's own as_of_months_ago/
    exit_months_ago CLI args already use, kept consistent rather than
    introducing a second date-math convention in the same codebase."""
    dates_desc = [today_date - pd.Timedelta(days=REBALANCE_MONTHS * 30 * i) for i in range(N_REBALANCES)]
    return sorted(dates_desc)


def _fetch_all(tickers: dict, workers: int) -> dict:
    """{ticker: raw_data}, one _fetch_raw_ticker_data call per ticker
    (ThreadPoolExecutor, network-bound -- same pattern as
    phase2_backtest.py's main()). Tickers whose fetch fails are
    dropped entirely, not retried -- same fail-open, report-don't-crash
    posture as every other per-ticker error in this project's backtest
    scripts."""
    raw_by_ticker = {}
    failed = []

    def _fetch(ticker):
        return ticker, _fetch_raw_ticker_data(ticker)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch, ticker) for ticker in tickers]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                ticker, raw_data = future.result()
                raw_by_ticker[ticker] = raw_data
            except Exception as e:
                failed.append(str(e))
            if completed % 25 == 0 or completed == len(tickers):
                print(f"  fetched {completed}/{len(tickers)} ({len(failed)} failed so far)", file=sys.stderr)

    if failed:
        print(f"  {len(failed)} tickers failed to fetch and are excluded from the whole run", file=sys.stderr)
    return raw_by_ticker


def _score_universe_at_date(raw_by_ticker: dict, categories: dict, as_of_date: pd.Timestamp,
                             market_history: pd.DataFrame, tnx_history: pd.DataFrame) -> dict:
    """{ticker: row} for every ticker that scores successfully as of
    as_of_date -- today_date is passed as as_of_date itself (not the
    real "today"), since this call only needs the point-in-time
    recommendation/composite_score for portfolio CONSTRUCTION, not
    _score_ticker_at_date's own realized_return_pct field (this
    script computes actual realized returns from the price path
    between rebalances directly, see _equity_segment, which is a
    truer measure than a single start/end scalar anyway)."""
    scores = {}
    for ticker, raw_data in raw_by_ticker.items():
        try:
            scores[ticker] = _score_ticker_at_date(
                ticker, categories[ticker], raw_data, as_of_date, as_of_date, market_history, tnx_history
            )
        except Exception:
            continue  # not investable at this date (e.g. no PIT financials yet) -- excluded, not an error
    return scores


def _top_n_buy_weights(scores: dict, top_n: int) -> dict:
    buys = [
        (ticker, row["composite_score"])
        for ticker, row in scores.items()
        if row.get("recommendation") == "Buy" and row.get("composite_score") is not None
    ]
    buys.sort(key=lambda t: t[1], reverse=True)
    top = buys[:top_n]
    if not top:
        return {}
    weight = 1.0 / len(top)
    return {ticker: weight for ticker, _ in top}


def _uniform_weights(tickers) -> dict:
    tickers = list(tickers)
    if not tickers:
        return {}
    weight = 1.0 / len(tickers)
    return {ticker: weight for ticker in tickers}


def _equity_segment(weights: dict, raw_by_ticker: dict, start_date: pd.Timestamp, end_date: pd.Timestamp,
                     calendar_index: pd.DatetimeIndex, starting_value: float):
    """Portfolio value path from start_date to end_date (inclusive),
    reindexed onto calendar_index (the S&P 500's own trading calendar,
    already fetched as market_history -- used as the shared reference
    calendar so every held ticker's price lands on the same dates,
    forward-filled across the rare missing day). {} weights (no
    Buy-rated tickers cleared the bar that quarter) returns a flat
    line at starting_value -- 100% cash for the period, a real and
    honestly-reportable outcome, not an error.

    Returns (equity_curve, drifted_weights_at_end) -- the second value
    is each held ticker's weight AFTER this window's price moves, NOT
    the target weight it started at. This matters even for a target
    that's literally unchanged rebalance to rebalance (the naive
    factor baseline holds the same universe every quarter): different
    tickers move by different amounts, so the ACTUAL weights drift
    away from equal-weight between rebalances, and correctly measuring
    next period's turnover means comparing the new target against
    these drifted weights, not against the stale prior target (which
    would report a naive equal-weight strategy as 0% turnover forever
    after the first rebalance -- confirmed as a real bug during this
    script's own smoke test before this fix)."""
    window = calendar_index[(calendar_index >= start_date) & (calendar_index <= end_date)]
    if len(window) == 0:
        window = pd.DatetimeIndex([start_date, end_date])

    if not weights:
        return pd.Series(starting_value, index=window), {}

    normalized_paths = {}
    for ticker, weight in weights.items():
        raw_data = raw_by_ticker.get(ticker)
        if raw_data is None:
            continue
        price_history = raw_data["price_history"]
        if price_history is None or price_history.empty:
            continue
        closes = price_history["Close"].reindex(window, method="ffill")
        if closes.isna().all():
            continue
        first_valid = closes.dropna().iloc[0]
        if not first_valid:
            continue
        normalized_paths[ticker] = (closes / first_valid) * weight

    if not normalized_paths:
        # Every held ticker was missing price data for this window
        # (shouldn't happen given it was scoreable at start_date, but
        # degrade to flat/cash rather than raise) -- same fail-open
        # posture as the rest of this script.
        return pd.Series(starting_value, index=window), {}

    portfolio_fraction = pd.concat(normalized_paths.values(), axis=1).sum(axis=1)
    # Any weight not covered by normalized_paths (a held ticker whose
    # price data dropped out mid-window) sits in cash for the
    # remainder, not silently vanishing from the portfolio's value.
    uncovered_weight = sum(weights.values()) - sum(
        w for t, w in weights.items() if t in normalized_paths
    )
    total_fraction = portfolio_fraction + uncovered_weight
    equity_curve = total_fraction * starting_value

    end_total = float(total_fraction.iloc[-1])
    drifted_weights = {}
    if end_total > 0:
        for ticker, path in normalized_paths.items():
            drifted_weights[ticker] = float(path.iloc[-1]) / end_total

    return equity_curve, drifted_weights


def _run_strategy(name: str, rebalance_dates: list, raw_by_ticker: dict, categories: dict,
                   market_history: pd.DataFrame, tnx_history: pd.DataFrame, weight_fn) -> dict:
    """Generic walk-forward runner -- used for both the real strategy
    (weight_fn ranks by composite_score) and the naive factor baseline
    (weight_fn is _uniform_weights over the whole universe), so both
    go through identical rebalance/cost/equity-curve mechanics and
    differ only in how a target portfolio is chosen."""
    calendar_index = market_history.index
    segments = []
    holdings_by_date = []
    turnovers = []
    costs = []
    portfolio_value = STARTING_CAPITAL
    previous_weights = {}

    for i, rebalance_date in enumerate(rebalance_dates):
        scores = _score_universe_at_date(raw_by_ticker, categories, rebalance_date, market_history, tnx_history)
        target_weights = weight_fn(scores, raw_by_ticker)

        period_turnover = pm.turnover(previous_weights, target_weights)
        cost = period_turnover * (TRANSACTION_COST_BPS / 10_000) * portfolio_value
        portfolio_value -= cost
        turnovers.append(period_turnover)
        costs.append(cost)
        holdings_by_date.append({"date": rebalance_date.date().isoformat(), "weights": target_weights,
                                  "n_holdings": len(target_weights)})

        if i + 1 < len(rebalance_dates):
            segment, drifted_weights = _equity_segment(
                target_weights, raw_by_ticker, rebalance_date, rebalance_dates[i + 1], calendar_index, portfolio_value
            )
            segments.append(segment if i == 0 else segment.iloc[1:])  # avoid double-counting the shared boundary date
            portfolio_value = float(segment.iloc[-1])
            # Drifted, not target -- see _equity_segment's own comment
            # on why the naive-baseline turnover bug this fixed
            # depended on exactly this distinction.
            previous_weights = drifted_weights

        print(f"  [{name}] {rebalance_date.date()}: {len(target_weights)} holdings, "
              f"turnover={period_turnover:.2f}, cost=${cost:,.0f}, value=${portfolio_value:,.0f}", file=sys.stderr)

    equity_curve = pd.concat(segments)
    period_values = [STARTING_CAPITAL] + [float(s.iloc[-1]) for s in segments]
    period_returns = pd.Series(period_values).pct_change().dropna()

    risk_free_rate = float(tnx_history["Close"].reindex(equity_curve.index, method="ffill").mean() / 100)

    metrics = {
        "cagr": pm.cagr(equity_curve),
        "annualized_volatility": pm.annualized_volatility(period_returns, periods_per_year=4),
        "sharpe_ratio": pm.sharpe_ratio(period_returns, risk_free_rate, periods_per_year=4),
        "sortino_ratio": pm.sortino_ratio(period_returns, risk_free_rate, periods_per_year=4),
        "max_drawdown": pm.max_drawdown(equity_curve),
        "hit_rate": pm.hit_rate(period_returns),
        "profit_factor": pm.profit_factor(period_returns),
        "avg_turnover_per_rebalance": float(np.mean(turnovers)),
        "total_transaction_costs": float(sum(costs)),
        "final_value": float(equity_curve.iloc[-1]),
    }

    return {
        "name": name,
        "metrics": metrics,
        "equity_curve": [{"date": d.date().isoformat(), "value": round(float(v), 2)} for d, v in equity_curve.items()],
        "holdings_by_date": holdings_by_date,
        "period_returns_pct": [round(r * 100, 2) for r in period_returns.tolist()],
    }


def _spy_buyhold(rebalance_dates: list, market_history: pd.DataFrame, tnx_history: pd.DataFrame) -> dict:
    """S&P 500, bought once at the first rebalance date and held with
    zero further trading -- the simplest possible comparator, using
    market_history (already fetched for every other purpose in this
    script, see MARKET_BENCHMARK) directly rather than a second fetch."""
    window = market_history.index[(market_history.index >= rebalance_dates[0]) & (market_history.index <= rebalance_dates[-1])]
    closes = market_history["Close"].reindex(window, method="ffill")
    equity_curve = (closes / closes.iloc[0]) * STARTING_CAPITAL

    quarter_boundaries = [d for d in rebalance_dates if d in equity_curve.index] or [equity_curve.index[0], equity_curve.index[-1]]
    period_values = equity_curve.reindex(pd.DatetimeIndex(quarter_boundaries), method="ffill")
    period_returns = period_values.pct_change().dropna()

    risk_free_rate = float(tnx_history["Close"].reindex(equity_curve.index, method="ffill").mean() / 100)

    metrics = {
        "cagr": pm.cagr(equity_curve),
        "annualized_volatility": pm.annualized_volatility(period_returns, periods_per_year=4),
        "sharpe_ratio": pm.sharpe_ratio(period_returns, risk_free_rate, periods_per_year=4),
        "sortino_ratio": pm.sortino_ratio(period_returns, risk_free_rate, periods_per_year=4),
        "max_drawdown": pm.max_drawdown(equity_curve),
        "hit_rate": pm.hit_rate(period_returns),
        "profit_factor": pm.profit_factor(period_returns),
        "avg_turnover_per_rebalance": 0.0,
        "total_transaction_costs": 0.0,
        "final_value": float(equity_curve.iloc[-1]),
    }
    return {
        "name": "spy_buyhold",
        "metrics": metrics,
        "equity_curve": [{"date": d.date().isoformat(), "value": round(float(v), 2)} for d, v in equity_curve.items()],
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-size", type=int, default=275)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def main():
    args = _parse_args()

    today_date = pd.Timestamp(datetime.utcnow().date())
    rebalance_dates = _rebalance_dates(today_date)

    print(f"Universe sample: {args.universe_size} tickers (sector-stratified, seed={UNIVERSE_SAMPLE_SEED})", file=sys.stderr)
    categories = _sample_universe(args.universe_size, UNIVERSE_SAMPLE_SEED)
    print(f"Rebalance dates ({len(rebalance_dates)}): {rebalance_dates[0].date()} -> {rebalance_dates[-1].date()}, "
          f"every {REBALANCE_MONTHS} months", file=sys.stderr)

    # 6y, not 5y (phase2_backtest.py's own default) -- the EARLIEST
    # rebalance date here is already YEARS_BACK in the past, and
    # _trailing_beta needs another full year of history ending AT that
    # date (BETA_WINDOW_TRADING_DAYS=252), so the fetch needs to reach
    # back YEARS_BACK+1 years minimum; 6y leaves real margin.
    print("Fetching market/risk-free-rate history...", file=sys.stderr)
    market_history = _tz_naive(yf.Ticker(MARKET_BENCHMARK).history(period="6y"))
    tnx_history = _tz_naive(yf.Ticker("^TNX").history(period="6y"))

    print(f"Fetching {len(categories)} tickers (one fetch each, reused across all {len(rebalance_dates)} rebalance dates)...",
          file=sys.stderr)
    raw_by_ticker = _fetch_all(categories, args.workers)
    print(f"{len(raw_by_ticker)}/{len(categories)} tickers fetched successfully", file=sys.stderr)

    print("\n=== Strategy: top-N Buy-rated by composite_score ===", file=sys.stderr)
    strategy_result = _run_strategy(
        "strategy", rebalance_dates, raw_by_ticker, categories, market_history, tnx_history,
        weight_fn=lambda scores, _raw: _top_n_buy_weights(scores, TOP_N_HOLDINGS),
    )

    print("\n=== Naive factor baseline: equal-weight the whole sampled universe ===", file=sys.stderr)
    naive_result = _run_strategy(
        "naive_factor_baseline", rebalance_dates, raw_by_ticker, categories, market_history, tnx_history,
        weight_fn=lambda _scores, raw: _uniform_weights(raw.keys()),
    )

    print("\n=== Comparator: S&P 500 buy-and-hold ===", file=sys.stderr)
    spy_result = _spy_buyhold(rebalance_dates, market_history, tnx_history)

    universe_tag = "ticker_universe_sample"
    output_path = str(SCRIPT_DIR / f"walkforward_results_{universe_tag}_{YEARS_BACK}y_quarterly.json")
    output = {
        "methodology": (
            f"Walk-forward portfolio backtest: {len(rebalance_dates)} rebalance dates every "
            f"{REBALANCE_MONTHS} months over {YEARS_BACK} years, long-only equal-weight top "
            f"{TOP_N_HOLDINGS} Buy-rated tickers by composite_score at each rebalance, "
            f"{TRANSACTION_COST_BPS}bps round-trip transaction cost on turnover, against a "
            f"sector-stratified {len(categories)}-ticker sample of scripts/ticker_universe.json."
        ),
        "universe_size": len(categories),
        "universe_fetched_ok": len(raw_by_ticker),
        "rebalance_dates": [d.date().isoformat() for d in rebalance_dates],
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "top_n_holdings": TOP_N_HOLDINGS,
        "starting_capital": STARTING_CAPITAL,
        "strategy": strategy_result,
        "naive_factor_baseline": naive_result,
        "spy_buyhold": spy_result,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "reproduce": "python scripts/walkforward_backtest.py",
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved -> {output_path}", file=sys.stderr)

    print("\n=== SUMMARY ===", file=sys.stderr)
    for result in (strategy_result, naive_result, spy_result):
        m = result["metrics"]
        cagr_str = f"{m['cagr']*100:.1f}%" if m["cagr"] is not None else "--"
        sharpe_str = f"{m['sharpe_ratio']:.2f}" if m["sharpe_ratio"] is not None else "--"
        dd_str = f"{m['max_drawdown']*100:.1f}%" if m["max_drawdown"] is not None else "--"
        print(f"  {result['name']:<25} CAGR={cagr_str:>8}  Sharpe={sharpe_str:>6}  MaxDD={dd_str:>8}  "
              f"Final=${m['final_value']:,.0f}", file=sys.stderr)


if __name__ == "__main__":
    main()
