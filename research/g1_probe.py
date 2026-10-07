"""g1_probe.py -- RESEARCH SANDBOX. G1 deep scan: for every ticker that was ever in the S&P 500 (2012-2026) but is not in our
price matrix, can yfinance return ANY history under that symbol?"""
import urllib.request, warnings, yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "data"
URL = "https://raw.githubusercontent.com/shardul0701/SP500-Survivorship-bias-data-2004-2026/main/src/sp500_ticker_history/sp500-ticker-changes-{y}.yaml"
ever = {}
for y in range(2012, 2027):
    d = yaml.safe_load(urllib.request.urlopen(URL.format(y=y), timeout=60).read())
    for t in d["tickers_on_Jan_1"]: ever.setdefault(t, y)
    for _, ch in (d.get("changes") or {}).items():
        for t in (ch.get("union") or []) + (ch.get("difference") or []): ever.setdefault(t, y)
adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
missing = sorted(t for t in ever if t not in adj.columns)
print(f"tickers ever in the S&P 500 since 2012: {len(ever)} | already in our price matrix: {len(ever)-len(missing)} | missing: {len(missing)}")
def probe(t):
    try:
        h = yf.Ticker(t).history(start="2008-01-01", auto_adjust=False)
        return t, len(h), (h.index.min().date().isoformat() if len(h) else None), (h.index.max().date().isoformat() if len(h) else None)
    except Exception:
        return t, 0, None, None
with ThreadPoolExecutor(8) as ex: res = list(ex.map(probe, missing))
df = pd.DataFrame(res, columns=["ticker", "rows", "first", "last"]); df.to_csv(DATA / "g1_probe.csv", index=False)
have = df[df.rows > 250]
print(f"yfinance returns >1 year of history for {len(have)} of {len(missing)} missing tickers ({len(have)/len(missing):.0%})")
print(f"   of those, history ends before 2025-01-01 (i.e. genuinely delisted names we could add): {(have['last'] < '2025-01-01').sum()}")
print(f"   zero rows: {(df.rows == 0).sum()} | tiny (<250 rows, likely a reused/reassigned symbol): {((df.rows > 0) & (df.rows <= 250)).sum()}")
print("examples recovered:", have[have['last'] < '2025-01-01'].head(12)[["ticker","first","last"]].values.tolist())
print("examples NOT recoverable:", df[df.rows == 0].head(15).ticker.tolist())
