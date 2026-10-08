"""
factor_ic.py -- RESEARCH SANDBOX (not production code).

P3 of SPRINT_TRACKER.md, implemented EXACTLY as pre-registered there (definitions,
dates, pass lines). Do not change a definition after seeing a result; add a new,
counted trial instead.

Run: python research/factor_ic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research import delisting  # noqa: E402
from research.pit_panel import annual_rows, load_facts  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

DEV_END = pd.Timestamp("2025-06-30")  # last rebalance date we may evaluate; later dates are the sealed holdout
FIRST_DATE = pd.Timestamp("2012-06-29")
MIN_NAMES = 100
FUND_ITEMS = ["net_income", "cfo", "capex", "total_assets", "shares_outstanding", "diluted_shares"]
MAX_STALENESS_DAYS = 550

FACTORS = ["mom_12_1", "low_vol", "earnings_yield", "fcf_yield", "roa"]
COMBOS = ["valuation_only", "multi_factor"]


def assert_development(date):
    if pd.Timestamp(date) > DEV_END:
        raise RuntimeError(f"{date} is in the sealed holdout (after {DEV_END.date()}); not allowed before Day 7")


def load_all():
    facts = load_facts()
    ann = annual_rows(facts[facts["logical"].isin(FUND_ITEMS)])
    ann = ann.sort_values(["cik", "logical", "end", "tag_rank", "filed"], ascending=[True, True, True, True, False])
    mapping = pd.read_csv(DATA / "ticker_cik.csv", dtype={"cik": str})
    mapping["bucket"] = mapping["category"].str.extract(r"\((S&P \d+)\)")
    splits = pd.read_parquet(DATA / "splits.parquet")
    return {
        "ann": ann,
        "map": mapping.drop_duplicates("ticker").set_index("ticker"),
        "adj": pd.read_parquet(DATA / "prices_adjclose.parquet"),
        "close": pd.read_parquet(DATA / "prices_close_splitadj.parquet"),
        "splits": {t: g.set_index("date")["ratio"].sort_index() for t, g in splits.groupby("ticker")},
    }


def latest_known(ann, as_of):
    """Latest fiscal-year value of each item that was public on `as_of` (best tag, latest filing <= as_of)."""
    known = ann[ann["filed"] <= as_of]
    known = known.drop_duplicates(["cik", "logical", "end"], keep="first")
    known = known.sort_values("end", ascending=False).drop_duplicates(["cik", "logical"], keep="first")
    known = known[(as_of - known["end"]).dt.days <= MAX_STALENESS_DAYS]
    return known.pivot(index="cik", columns="logical", values="val")


def split_factor_after(splits, tickers, date):
    out = {}
    for t in tickers:
        s = splits.get(t)
        out[t] = float(np.prod(s[s.index > date].values)) if s is not None and len(s) else 1.0
    return pd.Series(out)


def factors_at(data, date):
    """All five pre-registered factors for every ticker, using only information public before `date`."""
    adj, close = data["adj"], data["close"]
    pos = adj.index.get_loc(date)
    tickers = adj.columns
    stale = delisting.stale_tickers(date, tickers)  # delisted names carry their last price forward: never score from it
    p_now, p_m1, p_m12 = adj.iloc[pos].where(~stale), adj.iloc[pos - 21], adj.iloc[pos - 252]
    window = adj.iloc[pos - 252: pos + 1].pct_change(fill_method=None)
    window.loc[:, stale.reindex(window.columns).fillna(False).values.astype(bool)] = np.nan  # delisted: no volatility from carried prices

    out = pd.DataFrame(index=tickers)
    out["mom_12_1"] = (p_m1 / p_m12 - 1).where(p_now.notna())
    vol = window.std().where(window.notna().sum() >= 200)
    out["low_vol"] = (-vol).where(p_now.notna())

    mapping = data["map"].reindex(tickers)
    panel = latest_known(data["ann"], date - pd.Timedelta(days=1)).reindex(mapping["cik"].values)
    panel.index = tickers
    shares = panel["shares_outstanding"].fillna(panel["diluted_shares"])
    price_as_traded = close.iloc[pos] * split_factor_after(data["splits"], tickers, date).reindex(tickers)
    mcap = (price_as_traded * shares).where(lambda s: s > 0)
    # Dual-class companies share one CIK; EDGAR's cover-page share count is then ambiguous -> exclude from value factors.
    dual = mapping["cik"].duplicated(keep=False)
    out["earnings_yield"] = (panel["net_income"] / mcap).where(~dual)
    out["fcf_yield"] = ((panel["cfo"] - panel["capex"]) / mcap).where(~dual)
    out["roa"] = panel["net_income"] / panel["total_assets"]
    out["mcap"] = mcap
    out["bucket"] = mapping["bucket"]
    return out


def add_combos(f):
    r = f[FACTORS].rank(pct=True)
    value = r[["earnings_yield", "fcf_yield"]].mean(axis=1, skipna=True)
    f["valuation_only"] = value
    fam = pd.concat(
        [value.rank(pct=True), r["mom_12_1"], r["roa"], r["low_vol"]], axis=1
    )
    f["multi_factor"] = fam.mean(axis=1).where(fam.notna().sum(axis=1) >= 3)
    return f


def rank_ic(score, fwd, min_names=MIN_NAMES):
    ok = score.notna() & fwd.notna()
    if ok.sum() < min_names:
        return np.nan
    return score[ok].corr(fwd[ok], method="spearman")


def quintile_spread(score, fwd):
    ok = score.notna() & fwd.notna()
    if ok.sum() < MIN_NAMES:
        return np.nan
    q = pd.qcut(score[ok].rank(method="first"), 5, labels=False)
    return fwd[ok][q == 4].mean() - fwd[ok][q == 0].mean()


def quarter_ends(adj):
    idx = adj.index.to_series()
    return list(idx.groupby(idx.index.to_period("Q")).max())


def summarize(series):
    s = series.dropna()
    n = len(s)
    mean, std = s.mean(), s.std(ddof=1)
    return {"n": n, "mean_IC": mean, "std": std, "t": mean / (std / np.sqrt(n)) if n > 2 and std else np.nan,
            "hit": (s > 0).mean()}


def run():
    data = load_all()
    adj = data["adj"]
    qe = [d for d in quarter_ends(adj) if d >= FIRST_DATE]
    dates = [d for d in qe if d <= DEV_END]
    nxt = {d: qe[qe.index(d) + 1] for d in dates if qe.index(d) + 1 < len(qe)}

    # Sanity check on the split-corrected market cap before trusting any value factor.
    chk = factors_at(data, adj.index[adj.index <= "2013-06-28"][-1])
    print(f"sanity: AAPL market cap on 2013-06-28 = ${chk.loc['AAPL', 'mcap'] / 1e9:,.0f}B (expect roughly $400B)")

    rows, spreads, by_bucket = [], [], []
    for d in dates:
        assert_development(d)
        if d not in nxt:
            continue
        f = add_combos(factors_at(data, d))
        fwd = adj.loc[nxt[d]] / adj.loc[d] - 1
        rec = {"date": d}
        srec = {"date": d}
        for name in FACTORS + COMBOS:
            rec[name] = rank_ic(f[name], fwd)
            srec[name] = quintile_spread(f[name], fwd)
        rows.append(rec)
        spreads.append(srec)
        for b, g in f.groupby("bucket"):
            by_bucket.append({"date": d, "bucket": b,
                              **{n: rank_ic(g[n], fwd.reindex(g.index), min_names=25)
                                 for n in ("valuation_only", "multi_factor")}})
    ic = pd.DataFrame(rows).set_index("date")
    sp = pd.DataFrame(spreads).set_index("date")
    ic.to_csv(DATA / "p3_factor_ic_by_date.csv")

    print(f"\n=== P3 DEVELOPMENT RESULTS: {len(ic)} quarterly dates, {ic.index.min().date()} -> {ic.index.max().date()} ===")
    table = pd.DataFrame({n: summarize(ic[n]) for n in FACTORS + COMBOS}).T
    table["Q5-Q1 spread %/qtr"] = [sp[n].mean() * 100 for n in table.index]
    print(table.round(3).to_string())

    diff = (ic["multi_factor"] - ic["valuation_only"]).dropna()
    t_diff = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    print(f"\npaired difference multi_factor - valuation_only: mean IC diff = {diff.mean():+.4f}, t = {t_diff:+.2f} (n={len(diff)})")

    mf, vo = summarize(ic["multi_factor"]), summarize(ic["valuation_only"])
    alive = [n for n in FACTORS if summarize(ic[n])["mean_IC"] >= 0.02 and summarize(ic[n])["t"] >= 2]
    print("\nPRE-REGISTERED PASS LINES")
    print(f"  factors alive (mean IC >= 0.02 and t >= 2): {alive or 'none'}")
    improves = t_diff >= 2 and mf["mean_IC"] >= 0.03 and mf["t"] >= 2
    print(f"  multi_factor beats valuation_only (paired t >= 2): {t_diff >= 2}")
    print(f"  multi_factor mean IC >= 0.03 and t >= 2: {mf['mean_IC'] >= 0.03 and mf['t'] >= 2}")
    print(f"  => P3 'improves the model': {'PASS' if improves else 'FAIL'}")
    print(f"  (valuation_only on its own: mean IC {vo['mean_IC']:+.3f}, t {vo['t']:+.2f})")

    bb = pd.DataFrame(by_bucket)
    print("\n=== Descriptive split by index size bucket (not used to select anything) ===")
    for b, g in bb.groupby("bucket"):
        print(f"  {b}: valuation_only {summarize(g.set_index('date')['valuation_only'])['mean_IC']:+.3f} "
              f"(t {summarize(g.set_index('date')['valuation_only'])['t']:+.1f}) | multi_factor "
              f"{summarize(g.set_index('date')['multi_factor'])['mean_IC']:+.3f} "
              f"(t {summarize(g.set_index('date')['multi_factor'])['t']:+.1f})")

    # Regime view: the same IC averaged by year, to see stability rather than one pooled number.
    yearly = ic[["valuation_only", "multi_factor", "mom_12_1", "roa", "low_vol"]].groupby(ic.index.year).mean()
    print("\n=== Mean IC by year ===")
    print(yearly.round(3).to_string())
    return ic


if __name__ == "__main__":
    run()
