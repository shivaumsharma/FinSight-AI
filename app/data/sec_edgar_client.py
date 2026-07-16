"""
sec_edgar_client.py

Fetches real, ticker-agnostic company disclosures from SEC EDGAR's free
public API, replacing the hand-authored, closed-set transcript files
that used to live in app/data/transcripts/. Works for any ticker SEC
has a CIK for -- the overwhelming majority of US-listed public
companies -- instead of a handful of hardcoded demo files.

Source priority: the most recent 8-K's earnings-release exhibit
(Exhibit 99.x) is preferred over a 10-Q/10-K's Management's Discussion
& Analysis section. The 8-K exhibit is the document that actually
reads like "management discussing this quarter's results" (complete
with CEO/CFO quotes) -- a 10-Q/10-K's MD&A is long, dense regulatory
prose. 8-K submissions only list their own cover-page document in
SEC's per-company submissions feed, so finding the exhibit requires a
second lookup against that filing's own document index.

SEC requires a descriptive User-Agent identifying the requester on
every request (see https://www.sec.gov/os/webmaster-faq#developers) --
requests without one get blocked. Set SEC_EDGAR_CONTACT to override
the default used here.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "filings_cache"
TICKER_MAP_CACHE = CACHE_DIR / "ticker_cik_map.json"
TICKER_MAP_TTL_SECONDS = 7 * 24 * 3600  # SEC's ticker list changes rarely; refresh weekly

CONTACT = os.environ.get("SEC_EDGAR_CONTACT", "FinSight-AI research-tool sshivaum@gmail.com")
HEADERS = {"User-Agent": CONTACT}

# Text after any of these markers in an 8-K exhibit is legal
# boilerplate or raw financial-statement tables, not the narrative
# commentary RAG/sentiment actually want.
EXHIBIT_STOP_MARKERS = [
    "forward-looking statements",
    "condensed consolidated statements",
    "press contact:",
]


class SECEdgarClient:

    def __init__(self):
        CACHE_DIR.mkdir(exist_ok=True)
        self._ticker_to_cik = None

    # ------------------------------------------------------------
    # Ticker -> CIK
    # ------------------------------------------------------------

    def _load_ticker_map(self) -> Dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik

        if TICKER_MAP_CACHE.exists():
            age = time.time() - TICKER_MAP_CACHE.stat().st_mtime
            if age < TICKER_MAP_TTL_SECONDS:
                with open(TICKER_MAP_CACHE, "r", encoding="utf-8") as f:
                    self._ticker_to_cik = json.load(f)
                    return self._ticker_to_cik

        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()

        mapping = {
            entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
            for entry in raw.values()
        }

        with open(TICKER_MAP_CACHE, "w", encoding="utf-8") as f:
            json.dump(mapping, f)

        self._ticker_to_cik = mapping
        return mapping

    def get_cik(self, ticker: str) -> Optional[str]:
        return self._load_ticker_map().get(ticker.upper())

    # ------------------------------------------------------------
    # Filing discovery
    # ------------------------------------------------------------

    def _get_submissions(self, cik: str) -> dict:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def _filings_by_form(self, submissions: dict, form_type: str, limit: int, require_item: str = None) -> list:
        """
        Most recent `limit` filings of a given form type, newest
        first. For 8-Ks, `require_item` filters to those tagged with
        a specific SEC Item code -- "2.02" is "Results of Operations
        and Financial Condition" (i.e. an actual earnings
        announcement), as opposed to the many other reasons companies
        file 8-Ks (executive changes, governance/compensation-plan
        amendments, material agreements, etc.) that happen to also
        carry a numbered exhibit but aren't earnings-related.
        """
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        items = recent.get("items", [""] * len(forms))

        matches = []
        for i, form in enumerate(forms):
            if form != form_type:
                continue
            if require_item and require_item not in items[i].split(","):
                continue
            matches.append({
                "form": form,
                "accession_number": accessions[i],
                "filing_date": dates[i],
                "primary_document": primary_docs[i],
            })
            if len(matches) >= limit:
                break
        return matches

    def _find_exhibit_document(self, cik: str, accession_number: str) -> Optional[str]:
        """
        For an 8-K, the per-company submissions feed only lists the
        cover-page document, not its exhibits. The earnings press
        release (Exhibit 99.x) lives alongside it in the same
        accession -- found by listing that accession's own document
        index and matching filenames containing "ex99" (SEC exhibit
        files aren't consistently named, but this convention holds
        broadly enough to be worth trying before falling back to the
        cover page itself).
        """
        acc_nodash = accession_number.replace("-", "")
        cik_int = str(int(cik))

        resp = requests.get(
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/index.json",
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("directory", {}).get("item", [])

        for item in items:
            name = item.get("name", "").lower()
            if "ex99" in name or "ex-99" in name:
                return item["name"]

        return None

    # ------------------------------------------------------------
    # Document fetch + text extraction
    # ------------------------------------------------------------

    def _fetch_document(self, cik: str, accession_number: str, document: str) -> str:
        acc_nodash = accession_number.replace("-", "")
        cik_int = str(int(cik))
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{document}"
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        # Pass raw bytes to BeautifulSoup rather than resp.text -- its
        # UnicodeDammit encoding detection handles SEC's mix of UTF-8
        # and legacy Windows-1252 filings more reliably than requests'
        # own guess.
        return resp.content

    def _html_to_text(self, html_bytes) -> str:
        soup = BeautifulSoup(html_bytes, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def _trim_exhibit(self, text: str) -> str:
        """Cuts an 8-K exhibit down to its narrative body, dropping
        the legal boilerplate / raw financial tables that follow."""
        lowered = text.lower()
        cut_at = len(text)
        for marker in EXHIBIT_STOP_MARKERS:
            idx = lowered.find(marker)
            if idx != -1:
                cut_at = min(cut_at, idx)
        return text[:cut_at].strip()

    def _extract_mda_section(self, text: str) -> str:
        """
        For 10-Q/10-K filings, trims the (very long) full document
        down to the Management's Discussion & Analysis section --
        falls back to the first ~6000 characters if no MD&A heading
        is found, which is still far better than feeding an entire
        filing to the embedding model.

        "Management's Discussion and Analysis..." typically appears
        several times in a filing that isn't the actual section body:
        once in the table of contents (immediately followed by a page
        number), and often again as a cross-reference from elsewhere
        (e.g. Item 3 saying "as discussed in Item 2, Management's
        Discussion and Analysis..."). Neither position is reliable on
        its own, so instead of guessing by position, this picks
        whichever occurrence has the MOST text before the next "Item
        N" heading -- the real section has thousands of characters of
        prose before the next item; a TOC line or cross-reference has
        only a few dozen.
        """
        # Apostrophe-agnostic ("management.{0,2}s"): SEC filings render
        # the possessive with all sorts of characters depending on the
        # source encoding (straight quote, curly quote U+2019, or a
        # mangled replacement character) -- matching on any 0-2
        # characters between "management" and "s" sidesteps needing to
        # enumerate every variant.
        matches = list(re.finditer(r"management.{0,2}s\s+discussion\s+and\s+analysis", text, re.IGNORECASE))
        if not matches:
            return text[:6000]

        best_start, best_end, best_length = None, None, -1

        for match in matches:
            start = match.start()
            next_item = re.search(
                r"\bitem\s*[0-9]+[a-z]?\.\s",
                text[start + 50:],
                re.IGNORECASE,
            )
            end = start + 50 + next_item.start() if next_item else start + 12000
            if end - start > best_length:
                best_start, best_end, best_length = start, end, end - start

        return text[best_start:best_end]

    # ------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------

    def fetch_company_disclosure(self, ticker: str) -> Optional[dict]:
        """
        Returns {"text", "form", "filing_date", "source_url"} for the
        most useful recent disclosure available for `ticker`, or None
        if SEC has no CIK for it (e.g. some foreign private issuers
        file 20-F/6-K instead of 10-K/10-Q/8-K) or no qualifying
        filing was found. Disk-cached per ticker+accession so repeat
        queries don't re-hit SEC or re-parse HTML.
        """
        try:
            cik = self.get_cik(ticker)
            if not cik:
                return None

            submissions = self._get_submissions(cik)

            # Item 2.02 = "Results of Operations and Financial
            # Condition" -- SEC's own tag for an earnings-announcement
            # 8-K, as opposed to the many other reasons companies file
            # 8-Ks that happen to also carry a numbered exhibit (exec
            # changes, compensation-plan amendments, material
            # agreements, etc.) but aren't earnings-related.
            for filing in self._filings_by_form(submissions, "8-K", limit=8, require_item="2.02"):
                cached = self._read_cache(ticker, filing["accession_number"])
                if cached:
                    return cached

                exhibit = self._find_exhibit_document(cik, filing["accession_number"])
                if not exhibit:
                    continue

                html_bytes = self._fetch_document(cik, filing["accession_number"], exhibit)
                text = self._trim_exhibit(self._html_to_text(html_bytes))
                if not text.strip():
                    continue

                return self._write_cache(ticker, cik, filing, exhibit, text)

            # No recent 8-K had a usable exhibit -- fall back to the
            # most recent 10-Q, then 10-K, using their MD&A section.
            for form_type in ("10-Q", "10-K"):
                candidates = self._filings_by_form(submissions, form_type, limit=1)
                if not candidates:
                    continue
                filing = candidates[0]

                cached = self._read_cache(ticker, filing["accession_number"])
                if cached:
                    return cached

                document = filing["primary_document"]
                html_bytes = self._fetch_document(cik, filing["accession_number"], document)
                text = self._extract_mda_section(self._html_to_text(html_bytes))
                if not text.strip():
                    continue

                return self._write_cache(ticker, cik, filing, document, text)

            return None

        except (requests.RequestException, ValueError, KeyError):
            return None

    def _read_cache(self, ticker: str, accession_number: str) -> Optional[dict]:
        cache_file = CACHE_DIR / f"{ticker.upper()}_{accession_number}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_cache(self, ticker: str, cik: str, filing: dict, document: str, text: str) -> dict:
        result = {
            "text": text,
            "form": filing["form"],
            "filing_date": filing["filing_date"],
            "source_url": (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{filing['accession_number'].replace('-', '')}/{document}"
            ),
        }
        cache_file = CACHE_DIR / f"{ticker.upper()}_{filing['accession_number']}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f)
        return result
