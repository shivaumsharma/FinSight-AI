"""
delisted_build.py -- RESEARCH SANDBOX. G1: merge matched former S&P 500 members into the research data.

Inputs : research/data/delisted_map.csv (status AUTO rows only), research/data/tiingo/{TICKER}.parquet
Does   : (1) downloads EDGAR facts for the matched CIKs into the shard folder and rebuilds edgar_facts.parquet,
         (2) appends the tickers to ticker_cik.csv,
         (3) merges Tiingo prices into prices_close_splitadj / prices_adjclose / prices_volume / splits,
         (4) records each ticker's last real trading date (research/data/delisting_last_dates.json).
Safe to re-run: originals are backed up once as *_pre_delisted.*, and everything is rebuilt from the backups each time.

Run: python research/delisted_build.py
"""
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data.sec_edgar_client import HEADERS  # noqa: E402
from research import edgar_pit  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
TIINGO = DATA / "tiingo"
BACKUP_FILES = ["prices_close_splitadj.parquet", "prices_adjclose.parquet", "prices_volume.parquet", "splits.parquet",
                "ticker_cik.csv", "edgar_facts.parquet"]


def backup_once():
    for f in BACKUP_FILES:
        src = DATA / f
        dst = DATA / (src.stem + "_pre_delisted" + src.suffix)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def restore_from_backup():
    for f in BACKUP_FILES:
        src = DATA / f
        bak = DATA / (src.stem + "_pre_delisted" + src.suffix)
        if bak.exists():
            shutil.copy2(bak, src)


def tiingo_to_matrices(ticker):
    d = pd.read_parquet(TIINGO / f"{ticker}.parquet").sort_values("date").drop_duplicates("date").set_index("date")
    d = d[d["close"].notna() & (d["close"] > 0)]
    factor = d["splitFactor"].fillna(1.0)
    after = factor[::-1].cumprod()[::-1].shift(-1).fillna(1.0)          # product of splitFactor strictly AFTER each day
    close_splitadj = d["close"] / after                                  # split-adjusted (not dividend-adjusted), like yfinance Close
    ev = factor[factor != 1.0]
    splits = pd.DataFrame({"ticker": ticker, "date": ev.index, "ratio": ev.values})
    return close_splitadj, d["adjClose"], d["volume"], splits


def main():
    m = pd.read_csv(DATA / "delisted_map.csv")
    auto = m[(m.status == "AUTO") & m.cik.notna()].copy()
    auto = auto[[(TIINGO / f"{t}.parquet").exists() for t in auto.ticker]]
    auto["cik10"] = auto.cik.astype(int).map(lambda c: f"{c:010d}")
    print(f"{len(auto)} matched former members have prices")
    backup_once()
    restore_from_backup()

    # (1) EDGAR facts for the matched CIKs
    session = requests.Session()
    session.headers.update(HEADERS)
    edgar_pit.SHARDS.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for cik in auto.cik10.unique():
        shard = edgar_pit.SHARDS / f"{cik}.parquet"
        if shard.exists():
            continue
        data, status = edgar_pit._get_json(session, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        if data is not None:
            edgar_pit.extract_rows(data, cik).to_parquet(shard, index=False)
            fetched += 1
        time.sleep(edgar_pit.REQUEST_PAUSE_SECONDS)
    shards = sorted(edgar_pit.SHARDS.glob("*.parquet"))
    pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True).to_parquet(DATA / "edgar_facts.parquet", index=False)
    print(f"EDGAR: fetched {fetched} new filers; facts rebuilt from {len(shards)} shards")

    # (2) ticker -> CIK map (sector unknown for these; category keeps the S&P 500 bucket tag the loaders parse)
    base = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str})
    base = base[~base.ticker.isin(auto.ticker)]
    new = pd.DataFrame({"ticker": auto.ticker, "cik": auto.cik10, "category": "Unknown (S&P 500)"})
    pd.concat([base, new]).to_csv(DATA / "ticker_cik.csv", index=False)

    # (3) prices
    close, adj, vol = pd.read_parquet(DATA / "prices_close_splitadj.parquet"), pd.read_parquet(DATA / "prices_adjclose.parquet"), pd.read_parquet(DATA / "prices_volume.parquet")
    splits = pd.read_parquet(DATA / "splits.parquet")
    add_c, add_a, add_v, add_s, last_real = {}, {}, {}, [], {}
    for t in auto.ticker:
        c, a, v, s = tiingo_to_matrices(t)
        add_c[t], add_a[t], add_v[t] = c, a, v
        add_s.append(s)
        last_real[t] = str(c.index.max().date())
    idx = close.index
    new_c = pd.DataFrame(add_c).reindex(idx)
    new_a = pd.DataFrame(add_a).reindex(idx)
    new_v = pd.DataFrame(add_v).reindex(idx)
    # terminal return: carry the last real adjusted price forward (sold at the deal / last-trade price)
    for t in new_a.columns:
        last = pd.Timestamp(last_real[t])
        new_a.loc[new_a.index > last, t] = new_a[t].dropna().iloc[-1]
    keep = [c for c in close.columns if c not in new_c.columns]
    pd.concat([close[keep], new_c], axis=1).to_parquet(DATA / "prices_close_splitadj.parquet")
    pd.concat([adj[[c for c in adj.columns if c not in new_a.columns]], new_a], axis=1).to_parquet(DATA / "prices_adjclose.parquet")
    pd.concat([vol[[c for c in vol.columns if c not in new_v.columns]], new_v], axis=1).to_parquet(DATA / "prices_volume.parquet")
    pd.concat([splits[~splits.ticker.isin(auto.ticker)]] + add_s).to_parquet(DATA / "splits.parquet")
    (DATA / "delisting_last_dates.json").write_text(json.dumps(last_real, indent=1))
    print(f"prices merged: matrices now {pd.read_parquet(DATA / 'prices_adjclose.parquet').shape[1]} tickers; "
          f"{sum(len(s) for s in add_s)} split events added")


if __name__ == "__main__":
    main()
