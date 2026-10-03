"""survivorship.py -- RESEARCH SANDBOX. Measures how many point-in-time S&P 500 members our
current-constituent price universe is missing. Source: shardul0701/SP500-Survivorship-bias-data-2004-2026."""
import urllib.request
from pathlib import Path
import pandas as pd
import yaml

DATA = Path(__file__).resolve().parent / "data"
URL = ("https://raw.githubusercontent.com/shardul0701/SP500-Survivorship-bias-data-2004-2026/main/"
       "src/sp500_ticker_history/sp500-ticker-changes-{y}.yaml")

def jan1_members(year):
    with urllib.request.urlopen(URL.format(y=year), timeout=60) as r:
        return set(yaml.safe_load(r.read())["tickers_on_Jan_1"])

if __name__ == "__main__":
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    mapping = pd.read_csv(DATA / "ticker_cik.csv")
    universe = set(mapping["ticker"])
    for y in (2012, 2013, 2016, 2020, 2024, 2025):
        m = jan1_members(y)
        have_px = {t for t in m if t in adj.columns and adj[t].loc[:f"{y}-01-31"].notna().any()}
        in_univ = m & universe
        print(f"Jan 1 {y}: {len(m)} members | in our universe {len(in_univ)} ({len(in_univ)/len(m):.0%}) "
              f"| with price history at that date {len(have_px)} ({len(have_px)/len(m):.0%}) "
              f"| missing {len(m)-len(have_px)} ({1-len(have_px)/len(m):.0%})")
