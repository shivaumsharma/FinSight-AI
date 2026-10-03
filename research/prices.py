"""
prices.py -- RESEARCH SANDBOX (not production code).

Daily prices for the FinSight universe, back to 2008, for the factor tests (P3).
Three price series are kept because each factor needs a different one:
  - adj_close: split- AND dividend-adjusted (total return) -> momentum, volatility
  - close_splitadj: split-adjusted only -> times the cumulative LATER splits gives the
    as-traded price, which is what multiplies EDGAR's as-reported share counts to give
    a correct historical market cap (adj_close x as-reported shares would understate
    the market cap of any company that later split)
  - splits: the split events themselves

Survivorship caveat:
yfinance only has prices for tickers that still trade, so delisted names are
absent. That is a known, unfixable-with-free-data bias; the sprint measures its
size rather than pretending it away.

Run: python research/prices.py
Output (gitignored): research/data/prices_close_splitadj / prices_adjclose / prices_volume / splits .parquet
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UNIVERSE_PATH = ROOT.parent / "scripts" / "ticker_universe.json"
START = "2008-01-01"
WORKERS = 8


def fetch(ticker):
    for _ in range(2):
        try:
            hist = yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=True)
            if hist is not None and not hist.empty:
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                hist.index = hist.index.normalize()
                return ticker, hist[["Close", "Adj Close", "Volume", "Stock Splits"]]
        except Exception:
            continue
    return ticker, None


def main():
    DATA.mkdir(exist_ok=True)
    with open(UNIVERSE_PATH, encoding="utf-8") as f:
        tickers = list(json.load(f))
    closes, adj, volumes, splits, failed = {}, {}, {}, {}, []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, t) for t in tickers]
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, hist = fut.result()
            if hist is None:
                failed.append(ticker)
            else:
                closes[ticker] = hist["Close"]
                adj[ticker] = hist["Adj Close"]
                volumes[ticker] = hist["Volume"]
                sp = hist["Stock Splits"]
                splits[ticker] = sp[sp > 0]
            if i % 100 == 0 or i == len(tickers):
                print(f"  {i}/{len(tickers)} fetched ({len(failed)} failed)", file=sys.stderr)
    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).sort_index()
    close.to_parquet(DATA / "prices_close_splitadj.parquet")
    pd.DataFrame(adj).sort_index().to_parquet(DATA / "prices_adjclose.parquet")
    volume.to_parquet(DATA / "prices_volume.parquet")
    split_rows = [(t, d, r) for t, ser in splits.items() for d, r in ser.items()]
    pd.DataFrame(split_rows, columns=["ticker", "date", "ratio"]).to_parquet(DATA / "splits.parquet")
    pd.Series(failed, name="ticker").to_csv(DATA / "prices_failed.csv", index=False)
    print(f"\nprices for {close.shape[1]} tickers, {close.index.min().date()} -> {close.index.max().date()}; "
          f"{len(failed)} without price history", file=sys.stderr)


if __name__ == "__main__":
    main()
