"""
insider_pull.py -- RESEARCH SANDBOX. Downloads SEC's bulk Insider Transactions Data Sets (2012Q1-2025Q2 only; later quarters are
the sealed holdout period and are deliberately NOT fetched) and keeps open-market purchases by officers/directors.

Output (gitignored): research/data/insider_buys.parquet  columns: cik (10-digit str), filed, trans_date, owner, value
Run: python research/insider_pull.py        Resumable: finished quarters are cached as small shards.
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SHARDS = DATA / "insider_shards"
sys.path.insert(0, str(ROOT.parent))
from app.data.sec_edgar_client import HEADERS  # noqa: E402

URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"
QUARTERS = [(y, q) for y in range(2012, 2026) for q in (1, 2, 3, 4) if (y, q) <= (2025, 2)]


def parse(zf):
    read = lambda name: pd.read_csv(zf.open(name), sep="\t", dtype=str, low_memory=False, encoding="latin-1")
    sub = read("SUBMISSION.tsv")[["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"]]
    own = read("REPORTINGOWNER.tsv")[["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"]]
    tr = read("NONDERIV_TRANS.tsv")[["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"]]
    tr = tr[(tr.TRANS_CODE == "P") & (tr.TRANS_ACQUIRED_DISP_CD == "A")]
    sub = sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])]
    own = own[own.RPTOWNER_RELATIONSHIP.fillna("").str.contains("Officer|Director", case=False)]
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
    df["shares"] = pd.to_numeric(df.TRANS_SHARES, errors="coerce")
    df["price"] = pd.to_numeric(df.TRANS_PRICEPERSHARE, errors="coerce")
    df = df[(df.shares > 0) & (df.price > 0)]
    return pd.DataFrame({"cik": df.ISSUERCIK.str.zfill(10), "filed": pd.to_datetime(df.FILING_DATE, format="%d-%b-%Y", errors="coerce"),
                         "trans_date": pd.to_datetime(df.TRANS_DATE, format="%d-%b-%Y", errors="coerce"),
                         "owner": df.RPTOWNERCIK, "value": df.shares * df.price}).dropna(subset=["filed"])


def main():
    SHARDS.mkdir(parents=True, exist_ok=True)
    for y, q in QUARTERS:
        shard = SHARDS / f"{y}q{q}.parquet"
        if shard.exists():
            continue
        r = requests.get(URL.format(y=y, q=q), headers=HEADERS, timeout=300)
        if r.status_code != 200:
            print(f"{y}q{q}: HTTP {r.status_code} (skipped)", flush=True)
            continue
        parse(zipfile.ZipFile(io.BytesIO(r.content))).to_parquet(shard, index=False)
        print(f"{y}q{q}: ok ({len(r.content) / 1e6:.0f} MB)", flush=True)
        time.sleep(0.5)
    out = pd.concat([pd.read_parquet(f) for f in sorted(SHARDS.glob("*.parquet"))], ignore_index=True)
    out.to_parquet(DATA / "insider_buys.parquet", index=False)
    print(f"wrote {len(out):,} open-market purchases by officers/directors, {out.cik.nunique()} issuers, "
          f"{out.filed.min().date()} -> {out.filed.max().date()}")


if __name__ == "__main__":
    main()
