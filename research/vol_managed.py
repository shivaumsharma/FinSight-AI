"""vol_managed.py -- RESEARCH SANDBOX. A2 of the ALTERNATIVES PRE-REGISTRATION in SPRINT_TRACKER.md, exactly as written."""
import warnings
import numpy as np, pandas as pd, yfinance as yf
warnings.filterwarnings("ignore")

TARGET_VOL, WINDOW, COST = 0.15, 21, 0.0005
SPLIT = pd.Timestamp("2015-12-31")

def load():
    px = yf.Ticker("^GSPC").history(start="2006-01-01", auto_adjust=False)["Close"]; px.index = px.index.tz_localize(None).normalize()
    irx = yf.Ticker("^IRX").history(start="2006-01-01", auto_adjust=False)["Close"]; irx.index = irx.index.tz_localize(None).normalize()
    return px, (irx.reindex(px.index).ffill() / 100 / 252).fillna(0)

def run_strategy(px, cash, kind):
    ret = px.pct_change().fillna(0)
    me = px.groupby(px.index.to_period("M")).apply(lambda s: s.index[-1])           # month-end trading days
    rv = ret.rolling(WINDOW).std() * np.sqrt(252)
    sma = px.rolling(200).mean()
    expo_at = {}
    for d in me:
        if kind == "vol":  e = min(1.0, TARGET_VOL / rv.loc[d]) if pd.notna(rv.loc[d]) and rv.loc[d] > 0 else np.nan
        else:              e = (1.0 if px.loc[d] > sma.loc[d] else 0.0) if pd.notna(sma.loc[d]) else np.nan
        expo_at[d] = e
    e = pd.Series(expo_at).reindex(px.index).ffill().shift(1)       # decided at month-end close, applied from the next day
    start = e.first_valid_index(); e = e.loc[start:]
    turnover = e.diff().abs().fillna(0)
    r = e * ret.loc[start:] + (1 - e) * cash.loc[start:] - turnover * COST
    return r, e, ret.loc[start:], cash.loc[start:]

def stats(r, cash):
    ex = r - cash
    eq = (1 + r).cumprod()
    return {"CAGR": eq.iloc[-1] ** (252 / len(r)) - 1, "vol": r.std() * np.sqrt(252),
            "Sharpe": ex.mean() / r.std() * np.sqrt(252), "maxDD": (eq / eq.cummax() - 1).min()}

def main():
    px, cash = load(); rows = []; verdict = {}
    for kind, name in (("vol", "A2a vol-managed"), ("trend", "A2b trend filter")):
        r, e, bench, c = run_strategy(px, cash, kind)
        for lab, sl in (("2007-2015", slice(None, SPLIT)), ("2016-2025", slice(SPLIT + pd.Timedelta(days=1), None)), ("all", slice(None, None))):
            s, b = stats(r.loc[sl], c.loc[sl]), stats(bench.loc[sl], c.loc[sl])
            rows.append({"strategy": name, "period": lab, **{f"S_{k}": v for k, v in s.items()}, **{f"B_{k}": v for k, v in b.items()}, "avg_exposure": e.loc[sl].mean()})
        sub = [x for x in rows if x["strategy"] == name and x["period"] != "all"]
        verdict[name] = all(x["S_Sharpe"] >= x["B_Sharpe"] + 0.10 and abs(x["S_maxDD"]) <= 0.8 * abs(x["B_maxDD"]) for x in sub)
    df = pd.DataFrame(rows); pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False)); print("\nPRE-REGISTERED PASS (both halves, after costs):", verdict)

if __name__ == "__main__":
    main()
