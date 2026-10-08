"""tone_change_ic.py -- RESEARCH SANDBOX. Scores the G7 PRE-REGISTRATION in SPRINT_TRACKER.md exactly (P4's design, tone signals)."""
import re, sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import filing_change_ic as fc

DATA = fc.DATA
lm = pd.read_csv(DATA / "LM.csv", usecols=["Word", "Negative", "Uncertainty"])
NEG = set(lm.loc[lm.Negative > 0, "Word"]); UNC = set(lm.loc[lm.Uncertainty > 0, "Word"])
WORD = re.compile(r"[A-Za-z]+")

def shares(text):
    w = [x.upper() for x in WORD.findall(text)]
    n = max(len(w), 1)
    return sum(x in NEG for x in w) / n, sum(x in UNC for x in w) / n, len(w)

def load_pairs():
    rows, by = [], {}
    for p in sorted(fc.FILINGS.glob("*.txt")):
        t, d, _ = p.stem.split("_", 2); by.setdefault(t, []).append((pd.Timestamp(d), p))
    for t, items in by.items():
        prev = None
        for d, p in sorted(items):
            text = p.read_text(encoding="utf-8")
            if len(text) < fc.MIN_CHARS: prev = None; continue
            neg, unc, n = shares(text)
            if prev is not None and fc.GAP_DAYS[0] <= (d - prev[0]).days <= fc.GAP_DAYS[1]:
                rows.append({"ticker": t, "filed": d, "d_neg": neg - prev[1], "d_unc": unc - prev[2], "nwords": n})
            prev = (d, neg, unc)
    print(f"scored pairs: {len(rows)}"); return pd.DataFrame(rows)

def main():
    sample = pd.read_csv(DATA / "p4_sample.csv"); adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    adj = adj[[t for t in sample.ticker if t in adj.columns]]
    pairs = load_pairs(); pairs["nsf"] = 0.0                  # add_outcomes expects an 'nsf' column; unused here
    pairs = fc.add_outcomes(pairs, adj); pairs["group"] = pairs.filed.dt.to_period("Q")
    print(f"pairs with outcome: {len(pairs)} | median Item 1A words {int(pairs.nwords.median())}")
    print(f"mean LM-negative share change: {pairs.d_neg.mean()*100:+.3f} pts (std {pairs.d_neg.std()*100:.3f}); uncertainty: {pairs.d_unc.mean()*100:+.3f} pts")
    out = {}
    for name, col in (("T1 negative-tone change", "d_neg"), ("T2 uncertainty-tone change", "d_unc")):
        ics = {}
        for g, df in pairs.groupby("group"):
            if len(df) >= fc.MIN_GROUP: ics[str(g)] = (-df[col]).corr(df.abn_ret, method="spearman")
        s = pd.Series(ics); res = fc.summarize(list(s.values)); half = len(s) // 2
        a, b = s.iloc[:half].mean(), s.iloc[half:].mean()
        ok = res["groups"] >= 12 and res["mean_IC"] >= 0.02 and res["t"] >= 2.5 and a > 0 and b > 0
        out[name] = bool(ok)
        print(f"{name}: {res['groups']} groups | mean IC {res['mean_IC']:+.4f} | t {res['t']:+.2f} | hit {res['hit']:.0%} | halves {a:+.4f} / {b:+.4f} | pooled IC {(-pairs[col]).corr(pairs.abn_ret, method='spearman'):+.4f} | PASS: {bool(ok)}")
    print("\nPRE-REGISTERED VERDICT:", out)

if __name__ == "__main__":
    main()
