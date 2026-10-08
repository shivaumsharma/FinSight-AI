"""
filing_change_ic.py -- RESEARCH SANDBOX (not production code).

P4 of SPRINT_TRACKER.md, implemented EXACTLY as pre-registered there. Do not change a
definition after seeing a result; add a new, counted trial instead.

Signal: new_sentence_fraction of Item 1A vs the same company's previous 10-K.
Outcome: abnormal return from 2 trading days after filing to +63 trading days.
Metric: per calendar-quarter-of-filing group (>= 30 filings), Spearman IC of
(-new_sentence_fraction) vs abnormal return; mean IC, t across groups.

Run: python research/filing_change_ic.py   (after research/filing_text.py has finished)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FILINGS = DATA / "filings"

MIN_CHARS = 5000          # shorter than this = extraction failed, not a real section
MIN_WORDS = 8
GAP_DAYS = (300, 430)     # consecutive annual reports only
ENTRY_LAG = 2             # trading days after filing date
HORIZON = 63              # trading days
MIN_GROUP = 30
SEALED_AFTER = pd.Timestamp("2025-06-30")

SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WS_RE = re.compile(r"\s+")


def sentences(text):
    out = set()
    for s in SPLIT_RE.split(text):
        s = WS_RE.sub(" ", s.lower()).strip()
        if len(s.split()) >= MIN_WORDS:
            out.add(s)
    return out


def new_sentence_fraction(cur, prev):
    return len(cur - prev) / len(cur) if cur else np.nan


def load_pairs():
    rows, failed, total = [], 0, 0
    by_ticker = {}
    for p in sorted(FILINGS.glob("*.txt")):
        ticker, date, _ = p.stem.split("_", 2)
        by_ticker.setdefault(ticker, []).append((pd.Timestamp(date), p))
    for ticker, items in by_ticker.items():
        items.sort()
        prev = None
        for date, path in items:
            total += 1
            text = path.read_text(encoding="utf-8")
            if len(text) < MIN_CHARS:
                failed += 1
                prev = None  # chain is broken; do not pair across a gap
                continue
            cur = sentences(text)
            if prev is not None and GAP_DAYS[0] <= (date - prev[0]).days <= GAP_DAYS[1]:
                rows.append({"ticker": ticker, "filed": date, "nsf": new_sentence_fraction(cur, prev[1]),
                             "n_sent": len(cur)})
            prev = (date, cur)
    print(f"filings read: {total}, extraction failures: {failed}, scored pairs: {len(rows)}")
    return pd.DataFrame(rows)


def add_outcomes(pairs, adj):
    idx = adj.index
    # equal-weight market proxy: average return of all sample stocks over each window
    out = []
    for _, r in pairs.iterrows():
        if r["filed"] > SEALED_AFTER or r["ticker"] not in adj.columns:
            out.append(np.nan)
            continue
        pos = idx.searchsorted(r["filed"])
        e, x = pos + ENTRY_LAG, pos + ENTRY_LAG + HORIZON
        if x >= len(idx):
            out.append(np.nan)
            continue
        window = adj.iloc[[e, x]]
        rets = window.iloc[1] / window.iloc[0] - 1
        own = rets.get(r["ticker"])
        out.append(own - rets.mean() if pd.notna(own) else np.nan)
    pairs = pairs.copy()
    pairs["abn_ret"] = out
    return pairs.dropna(subset=["nsf", "abn_ret"])


def summarize(ics):
    s = pd.Series(ics).dropna()
    n = len(s)
    t = s.mean() / (s.std(ddof=1) / np.sqrt(n)) if n > 2 and s.std(ddof=1) else np.nan
    return {"groups": n, "mean_IC": s.mean(), "t": t, "hit": (s > 0).mean()}


def main():
    sample = pd.read_csv(DATA / "p4_sample.csv")
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    adj = adj[[t for t in sample["ticker"] if t in adj.columns]]
    pairs = add_outcomes(load_pairs(), adj)
    print(f"pairs with a usable outcome: {len(pairs)} across {pairs['ticker'].nunique()} companies")
    print(f"new_sentence_fraction: median {pairs['nsf'].median():.3f}, "
          f"p10 {pairs['nsf'].quantile(.1):.3f}, p90 {pairs['nsf'].quantile(.9):.3f}")

    pairs["group"] = pairs["filed"].dt.to_period("Q")
    ics, spreads, sizes = {}, {}, {}
    for g, df in pairs.groupby("group"):
        if len(df) < MIN_GROUP:
            continue
        score = -df["nsf"]
        ics[str(g)] = score.corr(df["abn_ret"], method="spearman")
        q = pd.qcut(score.rank(method="first"), 5, labels=False)
        spreads[str(g)] = df["abn_ret"][q == 4].mean() - df["abn_ret"][q == 0].mean()
        sizes[str(g)] = len(df)
    res = summarize(list(ics.values()))
    print(f"\n=== P4 PRIMARY SIGNAL (development): {res['groups']} filing-quarter groups with >= {MIN_GROUP} filings ===")
    print(f"mean IC {res['mean_IC']:+.4f} | t {res['t']:+.2f} | hit rate {res['hit']:.0%} | "
          f"Q5-Q1 abnormal return {np.mean(list(spreads.values())) * 100:+.2f}% over {HORIZON} trading days")
    print("per-group IC:", {k: round(v, 3) for k, v in ics.items()}, "\nsizes:", sizes)

    killed = not (res["mean_IC"] >= 0.02 and res["t"] >= 1.5)
    passed = res["mean_IC"] >= 0.02 and res["t"] >= 2
    print(f"\nPRE-REGISTERED: pass (IC >= 0.02 and t >= 2): {passed} | kill gate (IC < 0.02 or t < 1.5): {killed}")

    # Descriptive only (not a pass line): pooled across all filings
    pooled = (-pairs["nsf"]).corr(pairs["abn_ret"], method="spearman")
    print(f"descriptive pooled IC over all {len(pairs)} filings: {pooled:+.4f}")
    pairs.to_csv(DATA / "p4_pairs_scored.csv", index=False)


if __name__ == "__main__":
    main()
