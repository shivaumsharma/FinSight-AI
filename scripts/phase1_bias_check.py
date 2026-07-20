"""
phase1_bias_check.py

Phase 1 of the accuracy-measurement plan: a cheap, diagnostic sanity
check for whether the DCF-centric recommendation logic has a
structural bearish bias, independent of company quality.

Runs MarketDataTool + ValuationTool only (no LLM narrative, no RAG/
sentiment/news tools) across a deliberately diverse ticker set --
deep-value, mid-cap non-tech/finance, plausibly-overvalued, and
mega-cap -- and reports, per company: recommendation, DCF upside %,
relative valuation signal, and whether DCF and relative valuation
agree or disagree.

Not a backtest (see phase2_backtest.py for that) -- this only checks
today's calls against company *type*, to see whether Sell shows up
regardless of how cheap the company actually looks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.research_context import ResearchContext
from app.tools.market_data_tool import MarketDataTool
from app.tools.valuation_tool import ValuationTool
from app.reporting.report_data_builder import derive_recommendation

TICKERS = {
    # 2-3 deep-value / low-multiple (trade cheap on current cash flows)
    "XOM": "deep-value (energy)",
    "MO": "deep-value (tobacco)",
    "T": "deep-value (telecom)",

    # 2-3 mid-caps outside tech/finance
    "CMI": "mid-cap (industrials)",
    "DPZ": "mid-cap (consumer discretionary)",
    "DAL": "mid-cap (airlines/transport)",

    # 1-2 plausibly overvalued/distressed by consensus (Sell would be a correct call)
    "PLTR": "plausibly overvalued (high-multiple growth)",
    "CVNA": "plausibly overvalued/formerly distressed",

    # 1-2 more mega-caps for continuity with MSFT/BLK/GOOGL
    "AAPL": "mega-cap (continuity)",
    "NVDA": "mega-cap (continuity)",
}


def run_one(ticker: str):
    ctx = ResearchContext(ticker=ticker, question=f"Should I buy {ticker}?")

    MarketDataTool().run(ctx)
    ValuationTool().run(ctx)

    results = ctx.valuation_results or {}
    rec = derive_recommendation(results)
    relative = results.get("relative_valuation")

    rel_signal = relative["signal"] if relative else "unavailable"

    if relative is None:
        agreement = "n/a (no relative valuation data)"
    elif rec["rating"] == "Sell" and rel_signal == "expensive":
        agreement = "agree"
    elif rec["rating"] == "Sell" and rel_signal in ("cheap", "in-line"):
        agreement = "DISAGREE"
    elif rec["rating"] == "Buy" and rel_signal == "cheap":
        agreement = "agree"
    elif rec["rating"] == "Buy" and rel_signal in ("expensive", "in-line"):
        agreement = "DISAGREE"
    else:
        agreement = "n/a (Hold / Insufficient Data)"

    return {
        "ticker": ticker,
        "rating": rec["rating"],
        "upside_pct": results.get("upside_percent"),
        "relative_signal": rel_signal,
        "vs_history_pct": relative.get("vs_history_pct") if relative else None,
        "agreement": agreement,
    }


def main():
    rows = []
    for ticker, category in TICKERS.items():
        print(f"Running {ticker} ({category})...", file=sys.stderr)
        try:
            row = run_one(ticker)
            row["category"] = category
        except Exception as e:
            row = {
                "ticker": ticker, "category": category, "rating": f"ERROR: {e}",
                "upside_pct": None, "relative_signal": None,
                "vs_history_pct": None, "agreement": None,
            }
        rows.append(row)

    print()
    print(f"{'Ticker':<7}{'Category':<42}{'Rating':<10}{'Upside %':>10}  {'Rel. Signal':<12}{'vs Hist %':>10}  Agreement")
    print("-" * 120)
    for r in rows:
        upside = f"{r['upside_pct']:.1f}" if isinstance(r["upside_pct"], (int, float)) else "N/A"
        vs_hist = f"{r['vs_history_pct']:.1f}" if isinstance(r["vs_history_pct"], (int, float)) else "N/A"
        print(f"{r['ticker']:<7}{r['category']:<42}{str(r['rating']):<10}{upside:>10}  {str(r['relative_signal']):<12}{vs_hist:>10}  {r['agreement']}")

    valid = [r for r in rows if isinstance(r["rating"], str) and not r["rating"].startswith("ERROR")]
    sells = [r for r in valid if r["rating"] == "Sell"]
    buys = [r for r in valid if r["rating"] == "Buy"]
    holds = [r for r in valid if r["rating"] == "Hold"]

    print()
    print(f"Total run: {len(valid)}/{len(rows)}")
    print(f"Sell: {len(sells)}/{len(valid)} ({100*len(sells)/len(valid):.0f}%)")
    print(f"Hold: {len(holds)}/{len(valid)} ({100*len(holds)/len(valid):.0f}%)")
    print(f"Buy:  {len(buys)}/{len(valid)} ({100*len(buys)/len(valid):.0f}%)")

    deep_value_sells = [r for r in sells if "deep-value" in r["category"]]
    print(f"\nDeep-value names that still came out Sell: {len(deep_value_sells)}/3 "
          f"({', '.join(r['ticker'] for r in deep_value_sells) or 'none'})")


if __name__ == "__main__":
    main()
