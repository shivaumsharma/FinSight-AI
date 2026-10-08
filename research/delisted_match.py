"""
delisted_match.py -- RESEARCH SANDBOX. G1: link each former S&P 500 member (prices from Tiingo) to its SEC filer ID (CIK) by
company NAME, verifying every candidate, because ticker-based lookup is unsafe (symbols get reused: ALTR is now a different
company than the Altera that was in the index).

A candidate CIK is accepted automatically only if ALL hold:
  1. its name is similar to Tiingo's name for the ticker (sequence similarity >= 0.75, or one name's words contained in the
     other's: 'Denbury Resources' vs 'Denbury'),
  2. some 10-K was filed within ~15 months before / 4 months after the ticker's last actual price date (NOT "the last 10-K":
     acquired companies with public debt, like Avon, keep filing for years),
  3. cover-page shares x price implies a market cap between $0.4B and $1.5T at the 10-Ks filed DURING the years the ticker was
     in the index (median). If the SEC summary data has no share count (dual-class filers), this is "unknown" and the name
     similarity must be >= 0.9 instead,
  4. it filed 10-Ks across at least 70% of the years the ticker was actually IN the index, and
  5. its first 10-K is no later than one year after the ticker's first index year (a successor entity that took over the
     ticker, e.g. Allergan plc after Allergan Inc, fails here).
A match whose CIK is already in our universe is marked DUP (a second share class we already hold).
Everything else is marked REVIEW with the alternatives listed -- never guessed. Manual additions go in
research/data/delisted_overrides.csv (columns ticker,cik,reason); they are still run through checks 2-5.

Run: python research/delisted_match.py
Output: research/data/delisted_map.csv
"""
import difflib
import json
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.data.sec_edgar_client import HEADERS  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
TIINGO = DATA / "tiingo"
SUFFIX = re.compile(r"\b(incorporated|inc|corporation|corp|company|co|ltd|limited|plc|llc|lp|holdings?|group|the|class [a-z]|common stock|de|new)\b\.?", re.I)
MEMBERSHIP_URL = ("https://raw.githubusercontent.com/shardul0701/SP500-Survivorship-bias-data-2004-2026/main/src/"
                  "sp500_ticker_history/sp500-ticker-changes-{y}.yaml")


def membership_years():
    """ticker -> sorted list of calendar years it was an S&P 500 member (Jan-1 lists plus in-year additions/removals)."""
    cache = DATA / "sp500_membership_years.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import urllib.request

    import yaml

    years = {}
    for y in range(2012, 2027):
        d = yaml.safe_load(urllib.request.urlopen(MEMBERSHIP_URL.format(y=y), timeout=60).read())
        for t in d["tickers_on_Jan_1"]:
            years.setdefault(t, set()).add(y)
        for ch in (d.get("changes") or {}).values():
            for t in (ch.get("union") or []) + (ch.get("difference") or []):
                years.setdefault(t, set()).add(y)
    out = {t: sorted(v) for t, v in years.items()}
    cache.write_text(json.dumps(out))
    return out


def clean(name):
    name = re.sub(r"\(.*?\)|[^A-Za-z0-9& ]", " ", str(name))
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", name)).strip().lower()


def similarity(a, b):
    ca, cb = clean(a), clean(b)
    seq = difflib.SequenceMatcher(None, ca, cb).ratio()
    ta, tb = set(ca.split()), set(cb.split())
    contain = len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0
    return max(seq, 0.85 * contain)  # containment is capped below 0.9: it never clears the "unknown market cap" bar alone


def search(session, query):
    r = session.get("https://efts.sec.gov/LATEST/search-index", params={"keysTyped": query}, timeout=30)
    time.sleep(0.15)
    if r.status_code != 200:
        return []
    return [(int(h["_id"]), h["_source"].get("entity", "")) for h in r.json().get("hits", {}).get("hits", [])[:8]]


def companyfacts(session, cik):
    r = session.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", timeout=60)
    time.sleep(0.15)
    return r.json() if r.status_code == 200 else None


def profile(facts, price, member_years):
    """Last/first 10-K filing dates, filing years, and the median implied market cap at 10-Ks filed in member years."""
    rows = []
    for tax in ("us-gaap", "dei"):
        for tag, node in facts.get("facts", {}).get(tax, {}).items():
            for items in node.get("units", {}).values():
                rows += [(i["filed"], tag, i["val"]) for i in items if i.get("form") in ("10-K", "10-K/A") and "filed" in i]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["filed", "tag", "val"])
    df["filed"] = pd.to_datetime(df["filed"])
    caps = []
    if price is not None:
        sh = df[df.tag == "EntityCommonStockSharesOutstanding"].sort_values("filed").drop_duplicates("filed")
        for r in sh.itertuples():
            if member_years and r.filed.year not in member_years and (r.filed.year - 1) not in member_years:
                continue
            p = price[price.index <= r.filed]
            if len(p) and (r.filed - p.index[-1]).days < 10:
                caps.append(float(r.val * p.iloc[-1]))
    return {"last": df["filed"].max(), "first": df["filed"].min(), "years": set(df["filed"].dt.year),
            "dates": sorted(set(df["filed"])),
            "mcap": float(pd.Series(caps).median()) if caps else None}


def queries(name):
    c = clean(name)
    toks = [t for t in c.split() if len(t) >= 3]
    qs = {c}
    if toks:
        qs.add(toks[0])
        qs.add(max(toks, key=len))
    return [q for q in qs if q]


def main():
    meta = pd.read_csv(TIINGO / "_meta.csv")
    members = membership_years()
    # DUP means "already in the ORIGINAL (survivor) universe". Read the pre-merge backup if delisted_build.py has run before,
    # otherwise every previously added company would be mislabelled DUP of itself and silently dropped on the next build.
    original = DATA / "ticker_cik_pre_delisted.csv"
    known_ciks = set(int(c) for c in pd.read_csv(original if original.exists() else DATA / "ticker_cik.csv", dtype={"cik": str})["cik"])
    ov_path = DATA / "delisted_overrides.csv"
    overrides = pd.read_csv(ov_path, dtype={"cik": int}) if ov_path.exists() else pd.DataFrame(columns=["ticker", "cik", "reason"])
    session = requests.Session()
    session.headers.update(HEADERS)
    out = []
    for m in meta.itertuples():
        path = TIINGO / f"{m.ticker}.parquet"
        px = None
        if path.exists():
            px = pd.read_parquet(path).set_index("date")["close"].sort_index()
        yrs = members.get(m.ticker, [])
        if not isinstance(m.name, str) or not m.name.strip():
            out.append(dict(ticker=m.ticker, tiingo_name=None, status="NOMATCH", alternatives="no name from Tiingo"))
            print(f"{m.ticker:6s} NOMATCH  (no name)", flush=True)
            continue
        end = px.index.max() if px is not None and len(px) else pd.to_datetime(m.endDate)  # actual last price, not the metadata field
        if yrs:  # a symbol can outlive the company (fund wind-down, reused ticker): judge against the end of index membership
            end = min(end, pd.Timestamp(year=yrs[-1], month=12, day=31))
        cands, seen = [], set()
        for q in queries(m.name):
            for cik, ent in search(session, q):
                if cik not in seen:
                    seen.add(cik)
                    cands.append((cik, ent, False))
        for o in overrides[overrides.ticker == m.ticker].itertuples():
            if int(o.cik) not in seen:
                seen.add(int(o.cik))
                cands.append((int(o.cik), f"[override: {o.reason}]", True))
        scored = []
        for cik, ent, forced in cands:
            f = companyfacts(session, cik)
            if not f:
                continue
            ent = f.get("entityName", ent) if forced else ent
            sim = 1.0 if forced else similarity(m.name, ent)
            if sim < 0.45:
                continue
            p = profile(f, px, set(yrs))
            if p is None:
                continue
            gap = (end - p["last"]).days  # informational: days between the last 10-K ever filed and the last trading day
            ok_sim = sim >= 0.75
            ok_date = any(-120 <= (end - d).days <= 450 for d in p["dates"])
            cap = p["mcap"]
            ok_cap = (4e8 <= cap <= 1.5e12) if cap is not None else sim >= 0.9
            check_years = yrs[:-1] if len(yrs) > 1 else yrs  # the final member year's 10-K may never have been filed
            coverage = (sum(1 for y in check_years if y in p["years"] or y + 1 in p["years"]) / len(check_years)) if yrs else 0.0
            ok_start = bool(yrs) and p["first"].year <= min(yrs) + 1
            scored.append(dict(cik=cik, entity=ent, sim=round(sim, 2), last10k=p["last"].date().isoformat(), gap_days=gap,
                               mcap_bn=None if cap is None else round(cap / 1e9, 1), coverage=round(coverage, 2),
                               first10k=p["first"].date().isoformat(),
                               ok=ok_sim and ok_date and ok_cap and coverage >= 0.7 and ok_start))
        good = [s for s in scored if s["ok"]]
        best = max(good, key=lambda s: s["sim"]) if good else None
        unique = best is not None and sum(1 for s in good if s["cik"] != best["cik"] and s["sim"] >= best["sim"] - 0.05) == 0
        status = "AUTO" if best and unique else ("REVIEW" if scored else "NOMATCH")
        if status == "AUTO" and best["cik"] in known_ciks:
            status = "DUP"
        out.append(dict(ticker=m.ticker, tiingo_name=m.name, last_price=str(end)[:10], member_years=f"{yrs[0]}-{yrs[-1]}" if yrs else "",
                        status=status, cik=(best or {}).get("cik"), entity=(best or {}).get("entity"), sim=(best or {}).get("sim"),
                        mcap_bn=(best or {}).get("mcap_bn"), last10k=(best or {}).get("last10k"),
                        alternatives="; ".join(f"{s['entity']}[{s['cik']}] sim{s['sim']} first{s['first10k']} last{s['last10k']} cap{s['mcap_bn']} cov{s['coverage']}" for s in scored[:4])))
        print(f"{m.ticker:6s} {status:8s} {m.name[:34]:34s} -> {(best or {}).get('entity', '')[:34]}", flush=True)
    res = pd.DataFrame(out)
    res.to_csv(DATA / "delisted_map.csv", index=False)
    print("\n", res["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
