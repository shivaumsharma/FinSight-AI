"""
filing_text.py -- RESEARCH SANDBOX (not production code).

P4 of SPRINT_TRACKER.md, step 1: download 10-K filings for the pre-registered sample
and cut out Item 1A (Risk Factors). Text is cached under research/data/filings/.

Run: python research/filing_text.py [--limit N]
Resumable: cached filings are skipped.
"""

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.sec_edgar_client import HEADERS  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "filings"
SAMPLE_SIZE = 150
SEED = 0
FIRST_FILING = "2012-01-01"
LAST_FILING = "2025-06-30"  # later filings are the sealed holdout
PAUSE = 0.15  # SEC cap is 10 req/s across all workers; 3 workers x ~0.15s stays under it
WORKERS = 3

HEADING_GAP = r"[ \t]*"
START_RE = re.compile(r"(?:^|\n)" + HEADING_GAP + r"item\s*1a\s*[\.\-–—:]*\s*risk\s*factors", re.I)
END_RE = re.compile(
    r"\n" + HEADING_GAP + r"item\s*1b\s*[\.\-–—:]*\s*unresolved|\n" + HEADING_GAP + r"item\s*2\s*[\.\-–—:]*\s*properties",
    re.I,
)


def pick_sample():
    m = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str})
    m = m[m["category"].str.contains(r"S&P (?:400|600)", regex=True)]
    m = m[~m["cik"].duplicated(keep=False)]  # single-CIK companies only
    return m.sample(n=min(SAMPLE_SIZE, len(m)), random_state=SEED).sort_values("ticker").reset_index(drop=True)


def _get(session, url, as_json=False):
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.json() if as_json else r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def list_10ks(session, cik):
    """All 10-K filings (original only, not amendments) with filing dates in the registered window."""
    sub = _get(session, f"https://data.sec.gov/submissions/CIK{cik}.json", as_json=True)
    if not sub:
        return []
    blocks = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        extra = _get(session, f"https://data.sec.gov/submissions/{f['name']}", as_json=True)
        if extra:
            blocks.append(extra)
        time.sleep(PAUSE)
    rows = []
    for b in blocks:
        for form, acc, date, doc in zip(b["form"], b["accessionNumber"], b["filingDate"], b["primaryDocument"]):
            if form == "10-K" and FIRST_FILING <= date <= LAST_FILING:
                rows.append((acc, date, doc))
    return sorted(set(rows), key=lambda r: r[1])


def html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_item_1a(text):
    """Longest span between an 'Item 1A Risk Factors' heading and the next 1B/2 heading
    (the table of contents produces a short span; the real section the long one)."""
    best = ""
    for m in START_RE.finditer(text):
        end = END_RE.search(text, m.end())
        if not end:
            continue  # a heading with no following Item 1B/2 heading is not the section start
        span = text[m.end(): end.start()]
        if len(span) > len(best):
            best = span
    return best.strip()


def process(args):
    ticker, cik = args
    session = requests.Session()
    session.headers.update(HEADERS)
    n = 0
    for acc, date, doc in list_10ks(session, cik):
        path = OUT / f"{ticker}_{date}_{acc}.txt"
        if path.exists():
            continue
        html = _get(session, f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}")
        time.sleep(PAUSE)
        if not html:
            continue
        section = extract_item_1a(html_to_text(html))
        path.write_text(section, encoding="utf-8")  # empty file = extraction failed; kept so we don't retry forever
        n += 1
    return ticker, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pick_sample()
    sample.to_csv(DATA / "p4_sample.csv", index=False)
    jobs = list(zip(sample["ticker"], sample["cik"]))[: args.limit]
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for ticker, n in pool.map(process, jobs):
            done += 1
            print(f"  {done}/{len(jobs)} {ticker}: {n} new filings", file=sys.stderr)


if __name__ == "__main__":
    main()
