"""
tool_ablation.py -- RESEARCH SANDBOX. One common baseline, every candidate signal added on top, same data, same protocol.

Re-measures (on the FINAL survivorship-corrected development data; sealed holdout untouched) what each candidate "tool" changes:
  base            the A1b learned 4-quarter model (16 base features + sector dummies)
  +issuance       + net share issuance
  +earnings       + earnings surprise (SUE)
  +insider        + insider purchase value/market cap and number of insider buyers
  +all three
For each: 4-quarter rank IC (mean, Newey-West t), the paired change versus base (mean, Newey-West t), and the economic view:
top-decile long-only net excess return per quarter after 25 bps one-way cost.

These are the SAME definitions already counted as trials; re-running them on the final data adds none.
Run: python research/tool_ablation.py      Output: research/data/tool_ablation.json
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import insider_ic, sue_ic  # noqa: E402
from research import ml_cross_section as m  # noqa: E402

DATA, H, LAG, COST_BPS = m.DATA, 4, 3, 25
BASE_FEATS = list(m.FEATS)


def nw_t(x, lag=LAG):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def build():
    P, _ = m.build_panel()
    P = P.reset_index(drop=True)
    adj = pd.read_parquet(DATA / "prices_adjclose.parquet")
    idx = adj.index.to_series()
    qe = [d for d in sorted(idx.groupby(idx.index.to_period("Q")).max()) if d >= m.fi.FIRST_DATE]
    end_of = {d: qe[i + H] for i, d in enumerate(qe) if i + H < len(qe)}
    P["fwd4"] = [adj.at[end_of[d], t] / adj.at[d, t] - 1 if d in end_of and t in adj.columns else np.nan for d, t in zip(P.date, P.ticker)]
    P["y"] = P.groupby("date")["fwd4"].rank(pct=True)
    mp = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str}).drop_duplicates("ticker")
    by_cik = mp.groupby("cik")["ticker"].apply(list).to_dict()
    dates = sorted(P.date.unique())
    P = P.merge(sue_ic.sue_panel(dates, by_cik), on=["date", "ticker"], how="left")
    ins = insider_ic.insider_panel(dates, by_cik)
    P = P.merge(ins, on=["date", "ticker"], how="left")
    P["i_value"] = P.i_value.fillna(0.0)
    P["i_buyers"] = P.i_buyers.fillna(0)
    P["insider_i1"] = P.i_value / np.exp(P["log_mcap"])
    P["insider_i2"] = P.i_buyers.astype(float)
    return P, end_of


def walk_forward(P, end_of, feats):
    m.FEATS = feats
    X = m.prep(P)
    tests = [d for d in sorted(P.date.unique()) if d >= m.FIRST_TEST and d in end_of]
    preds = {k: pd.Series(np.nan, index=P.index) for k in ("hgb", "ridge")}
    model = {}
    for i, T in enumerate(tests):
        if i % m.RETRAIN_EVERY == 0:
            tr = P.date.map(lambda d: d in end_of and end_of[d] <= T) & P.y.notna()
            model["hgb"] = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200,
                                                         l2_regularization=1.0, random_state=0).fit(X[tr], P.loc[tr, "y"])
            model["ridge"] = Ridge(alpha=10).fit(X[tr], P.loc[tr, "y"])
        te = P.date == T
        for k in model:
            preds[k][te] = model[k].predict(X[te])
    return preds, tests


def evaluate(P, preds, tests, name):
    out = {}
    for k in ("hgb", "ridge"):
        ic = pd.Series({T: preds[k][P.date == T].corr(P.loc[P.date == T, "fwd4"], method="spearman")
                        for T in tests if ((P.date == T) & P.fwd4.notna()).sum() >= m.MIN_NAMES})
        rows, held_prev = [], set()
        for T in tests:
            g = P[(P.date == T) & P.fwd.notna()].assign(pred=preds[k])
            g = g[g.pred.notna()]
            if len(g) < m.MIN_NAMES:
                continue
            top = g.nlargest(max(10, len(g) // 10), "pred")
            held = set(top.ticker)
            turn = 1.0 if not held_prev else 1 - len(held & held_prev) / len(held)
            rows.append(top.fwd.mean() - turn * 2 * COST_BPS / 1e4 - g.fwd.mean())
            held_prev = held
        out[k] = {"ic": ic, "net_excess_q": np.array(rows)}
    return out


def main():
    P, end_of = build()
    cover = {"share_growth": P.share_growth.notna().mean(), "sue": P.sue.notna().mean(), "insider": (P.i_buyers > 0).mean()}
    print(f"panel rows {len(P)}; coverage of new features: " + ", ".join(f"{k} {v:.0%}" for k, v in cover.items()), flush=True)
    configs = {"base": [], "+issuance": ["share_growth"], "+earnings (SUE)": ["sue"],
               "+insider": ["insider_i1", "insider_i2"], "+all three": ["share_growth", "sue", "insider_i1", "insider_i2"]}
    results = {}
    for name, extra in configs.items():
        preds, tests = walk_forward(P, end_of, BASE_FEATS + extra)
        results[name] = evaluate(P, preds, tests, name)
        print(f"  done {name}", flush=True)
    rows = []
    for k in ("hgb", "ridge"):
        base_ic = results["base"][k]["ic"]
        for name in configs:
            ic = results[name][k]["ic"]
            diff = (ic - base_ic).dropna()
            ne = results[name][k]["net_excess_q"]
            rows.append({"model": k, "config": name, "IC_4q": ic.mean(), "NW_t": nw_t(ic),
                         "delta_vs_base": diff.mean() if name != "base" else 0.0,
                         "delta_NW_t": nw_t(diff) if name != "base" and diff.std() > 0 else float("nan"),
                         "net_excess_pts_per_qtr": ne.mean() * 100, "n_dates": len(ic)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    df.to_json(DATA / "tool_ablation.json", orient="records", indent=1)


if __name__ == "__main__":
    main()
