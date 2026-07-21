"""
build_ticker_universe.py

Builds a ~1000-ticker universe for a larger-scale phase2_backtest.py
run than the ~80-ticker hand-curated TICKERS dict -- more tickers
means less sampling noise in every accuracy/tuning question this
project has been asking (see scripts/tune_recommendation_config.py's
own docstring on this exact tradeoff).

Source: S&P 500 (large-cap) + S&P 400 (mid-cap) + first 100 of S&P 600
(small-cap), scraped from Wikipedia -- all 500 + all 400 + a 100-name
slice of 600 lands at exactly 1000, giving broad market-cap coverage
without needing a paid data source. Wikipedia blocks the default
urllib/requests user-agent (403), so a browser-like User-Agent header
is required -- unrelated to SEC EDGAR's OWN user-agent requirement
elsewhere in this project (app/data/*), just a separate site with the
same class of bot-blocking.

Category label per ticker is "{GICS Sector} (S&P {500,400,600})" --
coarser than the original TICKERS dict's hand-written labels
("deep-value (energy)" etc.), but that granularity doesn't scale to
1000 tickers by hand; sector + index tier is the natural substitute
and is exactly the axis the financials-accuracy finding needs (GICS
Sector "Financials" is a real, first-class field here, not something
inferred from a category string).

Output: scripts/ticker_universe.json ({ticker: category}).
"""

import json
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUTPUT_PATH = str(Path(__file__).resolve().parent / "ticker_universe.json")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

SOURCES = [
    ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500", None),
    ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400", None),
    ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "S&P 600", 100),
]


def _normalize_symbol(symbol: str) -> str:
    # yfinance expects "-" for share classes (e.g. "BRK-B"), Wikipedia
    # lists them with "." (e.g. "BRK.B") -- the single, well-known
    # gotcha with these lists.
    return symbol.strip().replace(".", "-")


def fetch_index(url: str, index_label: str, limit):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    if limit is not None:
        df = df.head(limit)
    return {
        _normalize_symbol(row["Symbol"]): f"{row['GICS Sector']} ({index_label})"
        for _, row in df.iterrows()
    }


def main():
    universe = {}
    for url, index_label, limit in SOURCES:
        print(f"Fetching {index_label}...", file=sys.stderr)
        index_map = fetch_index(url, index_label, limit)
        print(f"  {len(index_map)} tickers", file=sys.stderr)
        universe.update(index_map)  # later sources don't override earlier ones in practice (disjoint indices)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(universe, f, indent=2)

    print(f"\nTotal universe: {len(universe)} tickers -> {OUTPUT_PATH}", file=sys.stderr)

    sector_counts = {}
    for category in universe.values():
        sector = category.split(" (")[0]
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    print("\nBy sector:", file=sys.stderr)
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {sector:<30} {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
