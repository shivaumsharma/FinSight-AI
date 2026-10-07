"""ml_cross_section.py -- RESEARCH SANDBOX. A1 of the ALTERNATIVES PRE-REGISTRATION in SPRINT_TRACKER.md, exactly as written:
walk-forward learned cross-sectional model on the EDGAR panel. No tuning; two fixed models."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import delisting  # noqa: E402
from research import factor_ic as fi  # noqa: E402
from research.pit_panel import annual_rows, load_facts  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
FIRST_TEST, RETRAIN_EVERY, MIN_NAMES = pd.Timestamp("2016-06-30"), 4, 100
SCORES_FILE = "p2b_scores.csv"   # the holdout run points this at a file that also holds the holdout-date scores
ITEMS = ["net_income", "cfo", "capex", "total_assets", "revenue", "gross_profit", "equity", "long_term_debt",
         "debt_current", "shares_outstanding", "diluted_shares"]
FEATS = ["mom_12_1", "ret_1m", "vol", "log_mcap", "earnings_yield", "fcf_yield", "book_market", "roa", "gross_prof",
         "accruals", "asset_growth", "rev_growth", "leverage", "dcf_score", "relative_score", "upside_pct"]


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def known_two(ann, as_of):
    """Latest and prior fiscal-year value of every item, using only filings dated on or before as_of."""
    k = ann[ann["filed"] <= as_of].drop_duplicates(["cik", "logical", "end"], keep="first")
    k = k.sort_values("end", ascending=False)
    k["k"] = k.groupby(["cik", "logical"]).cumcount()
    k = k[(k["k"] <= 1) & ((as_of - k["end"]).dt.days <= fi.MAX_STALENESS_DAYS + (k["k"] * 365))]
    return k.pivot_table(index="cik", columns=["logical", "k"], values="val", aggfunc="first")


def build_panel():
    facts = load_facts()
    ann = annual_rows(facts[facts["logical"].isin(ITEMS)])
    ann = ann.sort_values(["cik", "logical", "end", "tag_rank", "filed"], ascending=[True, True, True, True, False])
    m = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str}).drop_duplicates("ticker").set_index("ticker")
    m["sector"] = m["category"].str.replace(r" \(S&P \d+\)", "", regex=True)
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    close = pd.read_parquet(DATA / "prices_close_splitadj.parquet")
    splits = pd.read_parquet(DATA / "splits.parquet")
    splits = {t: g.set_index("date")["ratio"].sort_index() for t, g in splits.groupby("ticker")}
    sc = pd.read_csv(DATA / SCORES_FILE, parse_dates=["date"])
    sc = sc[sc["error"].isna()]
    idx = adj.index.to_series()
    qe = [d for d in sorted(idx.groupby(idx.index.to_period("Q")).max()) if d >= fi.FIRST_DATE]
    nxt = {d: qe[i + 1] for i, d in enumerate(qe[:-1])}
    dual = m["cik"].duplicated(keep=False)
    parts = []
    for d in [x for x in qe if x <= fi.DEV_END]:
        pos = adj.index.get_loc(d)
        T = adj.columns
        stale = delisting.stale_tickers(d, T)
        p0, p1, p12 = adj.iloc[pos].where(~stale), adj.iloc[pos - 21], adj.iloc[pos - 252]
        win = adj.iloc[pos - 252: pos + 1].pct_change(fill_method=None)
        win.loc[:, stale.reindex(win.columns).fillna(False).values.astype(bool)] = np.nan  # delisted: no volatility from carried prices
        f = pd.DataFrame(index=T)
        f["mom_12_1"] = (p1 / p12 - 1).where(p0.notna())
        f["ret_1m"] = p0 / p1 - 1
        f["vol"] = win.std().where(win.notna().sum() >= 200)
        kt = known_two(ann, d - pd.Timedelta(days=1))
        mm = m.reindex(T)

        def g(it, k=0):
            if (it, k) not in kt.columns:
                return pd.Series(np.nan, index=T)
            return pd.Series(kt[(it, k)].reindex(mm["cik"].values).values, index=T)

        shares = g("shares_outstanding").fillna(g("diluted_shares"))
        later = pd.Series({t: float(np.prod(splits[t][splits[t].index > d].values)) if t in splits else 1.0 for t in T})
        mcap = (close.iloc[pos] * later * shares).where(lambda s: s > 0)
        ni, cfo, cap, at, at1 = g("net_income"), g("cfo"), g("capex"), g("total_assets"), g("total_assets", 1)
        ok = ~dual.reindex(T).fillna(False)
        ok = pd.Series(ok.values, index=T) if not isinstance(ok, pd.Series) else ok
        f["log_mcap"] = np.log(mcap).where(ok)
        f["earnings_yield"] = (ni / mcap).where(ok)
        f["fcf_yield"] = ((cfo - cap) / mcap).where(ok)
        f["book_market"] = (g("equity") / mcap).where(ok)
        f["roa"] = ni / at
        f["gross_prof"] = g("gross_profit") / at
        f["accruals"] = (ni - cfo) / at
        f["asset_growth"] = at / at1 - 1
        f["rev_growth"] = g("revenue") / g("revenue", 1) - 1
        f["leverage"] = (g("long_term_debt").fillna(0) + g("debt_current").fillna(0)) / at
        # net share issuance (A1c; NOT in FEATS, so the A1/A1b baselines are unchanged). A count change outside [-40%, +60%]
        # is almost certainly a split straddled by the two counts, not real issuance -> missing.
        sg = (shares / g("shares_outstanding", 1).fillna(g("diluted_shares", 1))) - 1
        f["share_growth"] = sg.where((sg > -0.4) & (sg < 0.6) & ok)
        s = sc[sc["date"] == d].set_index("ticker")[["dcf_score", "relative_score", "upside_pct"]]
        f = f.join(s)
        f["sector"] = mm["sector"]
        f["fwd"] = (adj.loc[nxt[d]] / adj.loc[d] - 1) if d in nxt else np.nan
        f = f[f["mom_12_1"].notna()].replace([np.inf, -np.inf], np.nan)
        f["date"] = d
        parts.append(f.reset_index(names="ticker"))
    return pd.concat(parts), nxt


def prep(df):
    X = df[FEATS].copy()
    X = X.groupby(df["date"].values).transform(lambda c: c.rank(pct=True) - 0.5).fillna(0.0)
    return pd.concat([X, pd.get_dummies(df["sector"], dtype=float)], axis=1)


def main():
    P, nxt = build_panel()
    P = P.reset_index(drop=True)
    P["y"] = P.groupby("date")["fwd"].rank(pct=True)
    X = prep(P)
    dates = sorted(P["date"].unique())
    tests = [d for d in dates if d >= FIRST_TEST and d in nxt]
    print(f"panel rows {len(P)}, dates {len(dates)}, test dates {len(tests)}, features {X.shape[1]}")
    preds = {"hgb": pd.Series(np.nan, index=P.index), "ridge": pd.Series(np.nan, index=P.index)}
    coefs, model = [], {}
    for i, T in enumerate(tests):
        if i % RETRAIN_EVERY == 0:
            tr = P["date"].map(lambda d: d in nxt and nxt[d] <= T) & P["y"].notna()
            model["hgb"] = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                                         l2_regularization=1.0, random_state=0).fit(X[tr], P.loc[tr, "y"])
            model["ridge"] = Ridge(alpha=10).fit(X[tr], P.loc[tr, "y"])
            coefs.append(pd.Series(model["ridge"].coef_, index=X.columns))
        te = P["date"] == T
        for k in model:
            preds[k][te] = model[k].predict(X[te])
    ic = {k: {} for k in preds}
    comp = {}
    for T in tests:
        g_ = P[P["date"] == T]
        for k in preds:
            ok = g_.index[g_["fwd"].notna() & preds[k][g_.index].notna()]
            if len(ok) >= MIN_NAMES:
                ic[k][T] = preds[k][ok].corr(g_.loc[ok, "fwd"], method="spearman")
        c_ = g_[["dcf_score", "relative_score", "fwd"]].dropna()
        if len(c_) >= MIN_NAMES:
            comp[T] = (0.8 * c_["dcf_score"] + 0.2 * c_["relative_score"]).corr(c_["fwd"], method="spearman")
    print(f"\n=== A1 out-of-sample, {len(tests)} test dates from {tests[0].date()} ===")
    for k in ("hgb", "ridge"):
        s = pd.Series(ic[k])
        print(f"  {k:6s} mean IC {s.mean():+.4f} | std {s.std():.3f} | t {tstat(s):+.2f} | positive {(s > 0).mean():.0%} | n={len(s)}"
              f" | PASS(IC>=0.03 & t>=2.5): {bool(s.mean() >= 0.03 and tstat(s) >= 2.5)}")
    c = pd.Series(comp)
    print(f"  production composite, same dates: mean IC {c.mean():+.4f} (t {tstat(c):+.2f})")
    s = pd.Series(ic["hgb"])
    print("  hgb IC by year:", {y: round(v, 3) for y, v in s.groupby(s.index.year).mean().items()})
    cm = pd.concat(coefs, axis=1).mean(axis=1).sort_values()
    print("\n  ridge mean coefficients (descriptive): " + ", ".join(f"{a}:{b:+.3f}" for a, b in list(cm.head(4).items()) + list(cm.tail(4).items())))
    P["pred_hgb"] = preds["hgb"]
    q = P[P["pred_hgb"].notna() & P["fwd"].notna()].copy()
    q["dec"] = q.groupby("date")["pred_hgb"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False))
    d = q.groupby(["date", "dec"])["fwd"].mean().unstack()
    print("  (descriptive) hgb decile mean next-quarter return %:", (d.mean() * 100).round(1).tolist(), f"| D9-D0 t {tstat(d[9] - d[0]):+.2f}")


if __name__ == "__main__":
    main()
