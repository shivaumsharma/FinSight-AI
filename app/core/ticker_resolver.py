"""
ticker_resolver.py

Small, deterministic (non-LLM) utilities for:

1. Mapping a ticker to its earnings-call transcript file.
2. Detecting which company/companies a free-text question refers to.

Why this is deterministic and NOT part of the LLM planner
-----------------------------------------------------------
Entity resolution (which ticker(s) is the user asking about) is a
lookup problem with a small, known universe of companies in this
project (the same five covered by app/benchmarks/*.json and
app/data/transcripts/*.txt). A keyword/regex lookup is faster,
free, and 100% reliable for this closed set, whereas asking an LLM
to "spell the ticker correctly" is a needless source of
hallucination. The LLM Planner's job is reasoning about *which
tools* to run, not entity extraction -- keeping the two concerns
separate makes both more reliable and easier to test.

This module previously existed only as an inline dict
(TRANSCRIPT_MAPPING) inside streamlit_app.py, and pointed at the
wrong path ("app/data/apple_q2.txt" instead of the real
"app/data/transcripts/apple_q2.txt"), so transcript loading for any
ticker silently failed with FileNotFoundError. That bug is fixed here.
"""

import re
from typing import List, Optional

# Ticker -> transcript file actually shipped with the project.
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

TRANSCRIPT_MAPPING = {
    "AAPL": BASE_DIR / "app" / "data" / "transcripts" / "apple_q2.txt",
    "MSFT": BASE_DIR / "app" / "data" / "transcripts" / "msft_q2.txt",
    "NVDA": BASE_DIR / "app" / "data" / "transcripts" / "nvda_q2.txt",
}
# Company name (and common aliases) -> ticker, covering the same
# universe as app/benchmarks/*.json.
COMPANY_NAME_TO_TICKER = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "tesla": "TSLA",
    "tsla": "TSLA",
}

_TICKER_TOKEN = re.compile(r"\b[A-Z]{2,5}\b")


def get_transcript_path(ticker: str):
    path = TRANSCRIPT_MAPPING.get(ticker.upper())
    return str(path) if path else None


def extract_companies(question: str, default_ticker: Optional[str] = None) -> List[str]:
    """
    Extracts an ordered, de-duplicated list of tickers mentioned in a
    question, matching both company names ("Apple", "Nvidia") and
    raw ticker tokens ("AAPL", "MSFT").

    If nothing is found and `default_ticker` is supplied (e.g. the
    ticker already selected in the UI), that is returned as a
    single-item list so callers always get at least one company.
    """

    found = []

    lowered = question.lower()
    for name, ticker in COMPANY_NAME_TO_TICKER.items():
        if name in lowered and ticker not in found:
            found.append(ticker)

    for token in _TICKER_TOKEN.findall(question):
        if token in COMPANY_NAME_TO_TICKER.values() and token not in found:
            found.append(token)

    if not found and default_ticker:
        found = [default_ticker.upper()]

    return found


def is_comparison_question(question: str) -> bool:
    """Heuristic for 'compare X and Y' style questions."""
    keywords = ["compare", " vs ", " vs.", "versus", "better buy", "which is a better"]
    lowered = f" {question.lower()} "
    return any(keyword in lowered for keyword in keywords)
