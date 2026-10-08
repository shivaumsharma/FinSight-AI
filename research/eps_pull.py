"""
eps_pull.py -- RESEARCH SANDBOX. Quarterly diluted EPS with filing dates, for the earnings-surprise (SUE) test.

Pulls SEC company facts for every CIK in ticker_cik.csv (resumable shard cache) and keeps:
  - 3-month EPS from 10-Qs (duration 80-100 days),
  - annual EPS from 10-Ks (340-390 days), used to derive Q4 = annual - (Q1+Q2+Q3).
Tag preference: EarningsPerShareDiluted, then EarningsPerShareBasicAndDiluted, then EarningsPerShareBasic.

Run: python research/eps_pull.py       Output: research/data/eps_quarterly.parquet  (cik, end, filed, eps, kind)
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SHARDS = DATA / "eps_shards"
sys.path.insert(0, str(ROOT.parent))
from app.data.sec_edgar_client import HEADERS  # noqa: E402
from research.edgar_pit import _get_json  # noqa: E402

TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"]


def extract(facts, cik):
    rows = []
    for rank, tag in enumerate(TAGS):
        node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
        if not node:
            continue
        for items in node.get("units", {}).values():
            for it in items:
                if it.get("form") not in ("10-Q", "10-Q/A", "10-K", "10-K/A") or "start" not in it:
                    continue
                rows.append((cik, rank, it["start"], it["end"], it["filed"], it["form"], it["val"]))
    df = pd.DataFrame(rows, columns=["cik", "rank", "start", "end", "filed", "form", "eps"])
    if df.empty:
        return df
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    days = (df["end"] - df["start"]).dt.days
    df["kind"] = pd.NA
    df.loc[days.between(80, 100) & df.form.str.startswith("10-Q"), "kind"] = "Q"
    df.loc[days.between(340, 390) & df.form.str.startswith("10-K"), "kind"] = "A"
    df = df[df.kind.notna()]
    # best tag, then the EARLIEST filing (what was known first); later restatements are not "surprises at the time"
    df = df.sort_values(["end", "kind", "rank", "filed"]).drop_duplicates(["end", "kind"], keep="first")
    return df[["cik", "end", "filed", "eps", "kind"]]


def main():
    SHARDS.mkdir(parents=True, exist_ok=True)
    ciks = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str})["cik"].drop_duplicates().tolist()
    session = requests.Session()
    session.headers.update(HEADERS)
    for i, cik in enumerate(ciks, 1):
        shard = SHARDS / f"{cik}.parquet"
        if shard.exists():
            continue
        data, _ = _get_json(session, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        extract(data, cik).to_parquet(shard, index=False) if data else pd.DataFrame(columns=["cik"]).to_parquet(shard, index=False)
        time.sleep(0.12)
        if i % 100 == 0:
            print(f"  {i}/{len(ciks)}", flush=True)
    frames = [pd.read_parquet(f) for f in SHARDS.glob("*.parquet")]
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out.to_parquet(DATA / "eps_quarterly.parquet", index=False)
    print(f"wrote {len(out):,} EPS rows for {out.cik.nunique()} companies "
          f"({(out.kind == 'Q').sum():,} quarterly, {(out.kind == 'A').sum():,} annual)")


if __name__ == "__main__":
    main()
