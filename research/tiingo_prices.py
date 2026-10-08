"""
tiingo_prices.py -- RESEARCH SANDBOX. G1: download daily prices for S&P 500 members that later left the index (delisted,
acquired, bankrupt) from Tiingo, so historical tests stop silently dropping the losers.

Setup: put your free Tiingo token in the project's .env as  TIINGO_API_KEY=...  (the token is read from the environment /
.env and is never printed or logged).

Run:
    python research/tiingo_prices.py check      # one request: is the key valid, and what do the free-tier limits look like
    python research/tiingo_prices.py run        # download every missing former member Tiingo lists (resumable)

Output (gitignored): research/data/tiingo/{TICKER}.parquet  with split-adjusted close, raw close, dividend, split factor.
Resumable: finished tickers are skipped. Rate limits (HTTP 429) are waited out, not retried in a loop.
"""
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "tiingo"
BASE = "https://api.tiingo.com/tiingo/daily"
START = "2008-01-01"


def load_key():
    key = os.environ.get("TIINGO_API_KEY")
    env = ROOT.parent / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("TIINGO_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("TIINGO_API_KEY is not set (add it to .env). Nothing was requested.")
    return key


def targets():
    """Former S&P 500 members (2012-2026) with no yfinance history that Tiingo's public ticker list covers."""
    probe = pd.read_csv(DATA / "g1_probe.csv")
    wanted = set(probe.loc[probe["rows"] == 0, "ticker"])
    listing = pd.read_csv(zipfile.ZipFile(DATA / "tiingo_supported_tickers.zip").open("supported_tickers.csv"))
    listing["ticker"] = listing["ticker"].str.upper()
    listing = listing[(listing["assetType"] == "Stock") & listing["ticker"].isin(wanted)].drop_duplicates("ticker")
    return listing[["ticker", "startDate", "endDate"]]


def fetch(session, ticker, key):
    url = f"{BASE}/{ticker.lower()}/prices"
    for attempt in range(14):
        r = session.get(url, params={"startDate": START, "format": "json", "token": key}, timeout=60)
        if r.status_code == 200:
            return pd.DataFrame(r.json())
        if r.status_code == 429:  # rate limit: wait it out rather than hammering
            wait = 70 if attempt < 2 else 600
            print(f"   rate limited on {ticker}; waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (401, 403):
            sys.exit(f"Tiingo rejected the token (HTTP {r.status_code}). Check TIINGO_API_KEY.")
        if r.status_code == 404:
            return None
        time.sleep(2 ** attempt)
    return None


def fetch_meta(session, ticker, key):
    """Company name and listing dates for a symbol (needed to match delisted companies to SEC filer IDs by NAME).
    Waits out rate limits like fetch() does."""
    for attempt in range(14):
        r = session.get(f"{BASE}/{ticker.lower()}", params={"token": key}, timeout=60)
        if r.status_code == 200:
            j = r.json()
            return {"ticker": ticker, "name": j.get("name"), "startDate": j.get("startDate"), "endDate": j.get("endDate"),
                    "exchange": j.get("exchangeCode")}
        if r.status_code == 429:
            wait = 70 if attempt < 2 else 600
            print(f"   rate limited on {ticker} (meta); waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        return None
    return None


def save_meta(rows):
    path = OUT / "_meta.csv"
    new = pd.DataFrame(rows)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=new.columns)
    pd.concat([old, new]).drop_duplicates("ticker", keep="last").to_csv(path, index=False)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("check", "run"):
        print(__doc__)
        return
    key = load_key()
    session = requests.Session()
    if mode == "check":
        r = session.get(f"{BASE}/aet/prices", params={"startDate": "2018-10-01", "endDate": "2018-11-28", "token": key}, timeout=60)
        print("HTTP", r.status_code)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            print(f"AET (Aetna, acquired 2018): {len(df)} daily rows, last row {df['date'].iloc[-1][:10]}  -> delisted data IS available")
        else:
            print(r.text[:300])
        for h in ("x-ratelimit-limit", "x-ratelimit-remaining", "retry-after"):
            if h in r.headers:
                print(f"{h}: {r.headers[h]}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    todo = targets()
    print(f"{len(todo)} former members to fetch")
    done = failed = 0
    for i, row in enumerate(todo.itertuples(), 1):
        path = OUT / f"{row.ticker}.parquet"
        meta_known = (OUT / "_meta.csv").exists() and row.ticker in set(pd.read_csv(OUT / "_meta.csv")["ticker"])
        if path.exists() and meta_known:
            continue
        if path.exists():  # prices already saved: only the name is missing
            m = fetch_meta(session, row.ticker, key)
            if m:
                save_meta([m])
            time.sleep(1.0)
            continue
        df = fetch(session, row.ticker, key)
        if df is None or df.empty:
            failed += 1
        else:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df.to_parquet(path, index=False)
            m = fetch_meta(session, row.ticker, key)
            if m:
                save_meta([m])
            done += 1
        if i % 10 == 0:
            print(f"  {i}/{len(todo)} ({done} saved, {failed} empty)", flush=True)
        time.sleep(1.0)
    print(f"finished: {done} saved, {failed} returned nothing")


if __name__ == "__main__":
    main()
