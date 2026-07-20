"""
investigate_narrative.py

Runs the REAL production pipeline (ResearchAgent.run(), same entry
point streamlit_app.py uses) end-to-end on a diverse set of tickers,
back to back, to gather actual data on two open questions:

1. How often does Executive Summary come out "Not available for this
   report", and what does the raw LLM output actually look like on
   those failures? (app/reporting/narrative_debug_log.py captures the
   full prompt + raw output + which headings matched, per call, in
   logs/narrative_debug.jsonl.)
2. Is grounding score correlated with prompt size, and is prompt size
   trending up as more report features (news, WACC-floor notes, etc.)
   have been added?

This script does NOT fix anything -- it only runs the pipeline and
prints a summary table at the end, cross-referencing narrative_debug
log entries against each run's final grounding_score.
"""

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.research_agent import ResearchAgent

TICKERS = [
    ("AAPL", "Should I buy Apple?"),
    ("MSFT", "Should I buy Microsoft?"),
    ("GOOGL", "Should I buy Google?"),
    ("NVDA", "Should I buy NVIDIA?"),
    ("XOM", "Should I buy Exxon?"),
    ("CMI", "Should I buy Cummins?"),
    ("PLTR", "Should I buy Palantir?"),
    ("JPM", "Should I buy JPMorgan?"),
]


def main():
    results = []
    for ticker, question in TICKERS:
        print(f"\n=== {ticker} ===", file=sys.stderr)
        start = time.time()
        try:
            context = ResearchAgent().run(question=question)
            elapsed = time.time() - start
            report_data = context.report_data or {}
            narrative = report_data.get("narrative", {})
            exec_summary = narrative.get("Executive Summary", "")
            results.append({
                "ticker": ticker,
                "elapsed_s": round(elapsed, 1),
                "grounding_score": context.evaluation.get("grounding_score"),
                "overall_score": context.evaluation.get("overall_score"),
                "citation_score": context.evaluation.get("citation_score"),
                "dcf_available": (context.valuation_results or {}).get("dcf_available"),
                "exec_summary_failed": exec_summary.strip() == "Not available for this report.",
                "exec_summary_len": len(exec_summary),
                "error": None,
            })
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            results.append({
                "ticker": ticker, "elapsed_s": round(time.time() - start, 1),
                "grounding_score": None, "overall_score": None, "citation_score": None,
                "dcf_available": None, "exec_summary_failed": None, "exec_summary_len": None,
                "error": str(e),
            })

    print("\n\n=== SUMMARY ===")
    header = f"{'Ticker':<7}{'Time(s)':>8}{'Grounding%':>12}{'Overall':>9}{'Citation%':>11}{'DCF?':<7}{'ExecSumFail':<13}{'ExecLen':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["error"]:
            print(f"{r['ticker']:<7} ERROR: {r['error']}")
            continue
        print(f"{r['ticker']:<7}{r['elapsed_s']:>8}{str(r['grounding_score']):>12}"
              f"{str(r['overall_score']):>9}{str(r['citation_score']):>11}"
              f"{str(r['dcf_available']):<7}{str(r['exec_summary_failed']):<13}{r['exec_summary_len']:>8}")


if __name__ == "__main__":
    main()
