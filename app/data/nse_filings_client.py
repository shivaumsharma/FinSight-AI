"""
nse_filings_client.py

Fetches real, ticker-agnostic corporate disclosures from NSE India for
.NS-listed tickers -- the NSE analog of sec_edgar_client.py's SEC EDGAR
client, matching its exact public shape (fetch_company_disclosure(ticker)
-> Optional[dict] with the same {"text", "form", "filing_date",
"accession_number", "source_url"} keys) so everything downstream
(RAGPipeline, CitationEngine, Management Sentiment) needs zero changes
to accept NSE evidence alongside SEC evidence -- see rag_pipeline.py's
ingest_company_disclosure for the one branch that picks which client to
call.

Endpoint discovered via a live manual spike (not guessed), against NSE's
real corporate-announcements feed:
  GET https://www.nseindia.com/api/corporate-announcements
      ?index=equities&symbol={SYMBOL}&from_date={DD-MM-YYYY}&to_date={DD-MM-YYYY}
Confirmed live: works with a browser-like User-Agent + Accept +
Referer + X-Requested-With headers alone -- no cookie warm-up needed
for this specific endpoint (unlike nseindia.com's own HTML pages, which
returned 403 in the same spike even with a browser User-Agent -- this
API path sits behind a lighter bot-detection tier). Still built on a
requests.Session() rather than a bare requests.get, so adding a warm-up
GET later (if NSE's protection tightens) is a one-line addition, not a
rewrite.

Each announcement's PDF attachment (attchmntFile) is hosted on the
*separate* nsearchives.nseindia.com subdomain, which -- also confirmed
live -- needs the browser User-Agent + Referer (the plain identifying
User-Agent that works for archives.nseindia.com's static CSV in
company_resolver.py times out here).

"Outcome of Board Meeting" / "Financial Result Updates" are NSE's own
`desc` categories for a quarterly-results announcement -- confirmed
against real filings to be the analog of SEC's 8-K Item 2.02 exhibit
(a results letter to the exchange, PDF attached, with real narrative
text on its cover page, not just a numbers table).

Real, load-bearing uncertainty this client must survive, not guess past:
this exact class of failure (NSE blocking cloud-datacenter/CI egress
IPs) is already documented twice elsewhere in this codebase --
archives.nseindia.com blocking GitHub Actions (see the excluded
test_company_resolver.py), and market_movers.py's own docstring on
Yahoo Finance throttling Cloud Run's egress IP the same way. The spike
behind this client succeeded from a local dev machine; there is no
guarantee it succeeds from Cloud Run. fetch_company_disclosure must
therefore NEVER raise past its own broad except -- same None-on-any-
failure contract SECEdgarClient already has -- and every failure is
logged at WARNING so a degraded-evidence deploy is observable rather
than silently invisible.
"""

import io
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pypdf
import requests

from app.core.retry import retry_on_transient_error

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "filings_cache"

ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"

# A real browser User-Agent, not an identifying one -- see this module's
# own docstring: the identifying UA sec_edgar_client.py's HEADERS uses
# (and that archives.nseindia.com accepts fine) times out against both
# the live announcements API and nsearchives.nseindia.com's PDF host.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}
_PDF_HEADERS = {
    "User-Agent": _BROWSER_HEADERS["User-Agent"],
    "Referer": "https://www.nseindia.com/",
}

# Wide enough to comfortably contain the latest quarter's results
# announcement even with real-world reporting lag (companies don't
# always report within 45 days of quarter-end), narrow enough to keep
# the response reasonably sized -- a ticker with no date filter at all
# returns its ENTIRE announcement history (confirmed live: 3335 rows
# for a single large-cap), which is both slow and mostly irrelevant.
LOOKBACK_DAYS = 200

# NSE's own category labels for a quarterly-results announcement,
# confirmed against real filings (see this module's docstring) --
# matched case-insensitively as a substring of the `desc` field, not an
# exact-set membership check, since NSE's own vocabulary has drifted
# across years ("Financial Result Updates" vs. "Financial Results").
_RESULTS_DESC_PATTERN = re.compile(r"board meeting|financial result", re.IGNORECASE)

# Caps total extracted PDF text -- these filings run to dozens of pages
# once the full audited financial statements and notes are attached;
# the cover letter + results summary (what RAG/Management Sentiment
# actually want, same "narrative over raw tables" preference
# sec_edgar_client.py's EXHIBIT_STOP_MARKERS encodes for SEC exhibits)
# is always in the first several pages. Same order of magnitude as
# sec_edgar_client.py's own MD&A fallback cap (12000 chars).
MAX_EXTRACTED_CHARS = 15000
MAX_PDF_PAGES = 10


_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """
    A shared requests.Session rather than bare requests.get calls --
    confirmed live (see this module's docstring) that neither the
    announcements API nor the PDF host currently needs a cookie
    warm-up, but a Session is what makes adding one later (if NSE's
    bot-detection tightens) a single `session.get("https://www.nseindia.com/")`
    line rather than a rewrite of every call site.
    """
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _fetch_json(url: str, params: dict) -> object:
    def _do():
        resp = _get_session().get(url, headers=_BROWSER_HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp

    return retry_on_transient_error(_do).json()


def _fetch_pdf_bytes(url: str) -> bytes:
    def _do():
        resp = _get_session().get(url, headers=_PDF_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp

    return retry_on_transient_error(_do).content


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    total_len = 0
    for page in reader.pages[:MAX_PDF_PAGES]:
        text = page.extract_text() or ""
        parts.append(text)
        total_len += len(text)
        if total_len >= MAX_EXTRACTED_CHARS:
            break
    return "\n".join(parts)[:MAX_EXTRACTED_CHARS]


def _parse_announcement_date(an_dt: str) -> str:
    """"07-Aug-2026 17:07:35" -> "2026-08-07" -- ISO date, matching
    SEC's own filing_date format (sec_edgar_client.py's filing_date
    comes straight from SEC's already-ISO submissions JSON)."""
    return datetime.strptime(an_dt, "%d-%b-%Y %H:%M:%S").date().isoformat()


def _read_cache(ticker: str, accession_number: str) -> Optional[dict]:
    cache_file = CACHE_DIR / f"{ticker.upper()}.NS_{accession_number}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_cache(ticker: str, result: dict) -> dict:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker.upper()}.NS_{result['accession_number']}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


def fetch_company_disclosure(ticker: str) -> Optional[dict]:
    """
    Returns {"text", "form", "filing_date", "accession_number",
    "source_url"} for the most recent quarterly-results announcement
    NSE has for `ticker` (a bare NSE symbol, e.g. "RELIANCE" -- callers
    strip the ".NS" suffix before calling, see rag_pipeline.py), or
    None if NSE has no qualifying announcement, the ticker isn't found,
    or ANY step of this fails for any reason (network, parsing, a
    malformed response) -- never raises. See this module's own
    docstring for why that contract is load-bearing here, not optional.
    """
    try:
        today = datetime.now()
        data = _fetch_json(ANNOUNCEMENTS_URL, params={
            "index": "equities",
            "symbol": ticker.upper(),
            "from_date": (today - timedelta(days=LOOKBACK_DAYS)).strftime("%d-%m-%Y"),
            "to_date": today.strftime("%d-%m-%Y"),
        })

        if not isinstance(data, list):
            return None

        results_announcements = [
            item for item in data
            if isinstance(item, dict) and _RESULTS_DESC_PATTERN.search(item.get("desc") or "")
            and item.get("attchmntFile") and item.get("seq_id") and item.get("an_dt")
        ]
        if not results_announcements:
            return None

        # Most recent first -- sort_date is NSE's own sortable
        # "YYYY-MM-DD HH:MM:SS" field, more reliable to sort on than
        # re-parsing an_dt's "DD-Mon-YYYY" text a second time here.
        results_announcements.sort(key=lambda item: item.get("sort_date") or "", reverse=True)
        latest = results_announcements[0]

        accession_number = str(latest["seq_id"])
        cached = _read_cache(ticker, accession_number)
        if cached:
            return cached

        pdf_bytes = _fetch_pdf_bytes(latest["attchmntFile"])
        text = _extract_pdf_text(pdf_bytes)
        if not text.strip():
            # A scanned filing with no text layer -- out of scope this
            # pass (no OCR), same "degrade rather than return junk"
            # choice sec_edgar_client.py makes when an exhibit/MD&A
            # extraction comes back empty.
            return None

        result = {
            "text": text,
            "form": latest.get("desc") or "NSE Corporate Announcement",
            "filing_date": _parse_announcement_date(latest["an_dt"]),
            "accession_number": accession_number,
            "source_url": latest["attchmntFile"],
        }
        return _write_cache(ticker, result)

    except Exception as e:
        logger.warning(f"NSE filing fetch failed for {ticker}.NS (degrading to no evidence): {e}")
        return None


class NSEFilingsClient:
    """
    Thin class wrapper around the module-level fetch_company_disclosure
    above, purely so rag_pipeline.py can hold `self.nse_client =
    NSEFilingsClient()` alongside `self.sec_client = SECEdgarClient()`
    and call both the same way (`.fetch_company_disclosure(ticker)`) --
    unlike SECEdgarClient, this client has no real per-instance state
    (no ticker->CIK map to cache), so the module-level function is
    where the actual logic and unit tests live.
    """

    def fetch_company_disclosure(self, ticker: str) -> Optional[dict]:
        return fetch_company_disclosure(ticker)
