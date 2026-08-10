"""
Unit tests for app/data/nse_filings_client.py -- hand-built fixtures,
monkeypatched at the same seams sec_edgar_client.py's own network calls
would be mocked at (the module's own _fetch_json/_fetch_pdf_bytes/
_extract_pdf_text helpers), no live network. Matches this session's own
hard-won lesson from the CI fix: a real NSE-dependent test blocked on
GitHub Actions' IP ranges (test_company_resolver.py, now excluded from
CI) -- this file must never make a real request.
"""

import json

import pytest

from app.data import nse_filings_client as nse


def _announcement(desc="Outcome of Board Meeting", seq_id="1", sort_date="2026-07-17 19:07:44",
                   an_dt="17-Jul-2026 19:07:44", attchmnt_file="https://nsearchives.nseindia.com/x.pdf"):
    return {
        "an_dt": an_dt,
        "attchmntFile": attchmnt_file,
        "attchmntText": "some cover text",
        "desc": desc,
        "seq_id": seq_id,
        "sort_date": sort_date,
        "symbol": "RELIANCE",
    }


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(nse, "CACHE_DIR", tmp_path)


def test_fetch_company_disclosure_success(monkeypatch):
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [_announcement()])
    monkeypatch.setattr(nse, "_fetch_pdf_bytes", lambda url: b"fake-pdf-bytes")
    monkeypatch.setattr(nse, "_extract_pdf_text", lambda pdf_bytes: "Unaudited results for the quarter.")

    result = nse.fetch_company_disclosure("RELIANCE")

    assert result == {
        "text": "Unaudited results for the quarter.",
        "form": "Outcome of Board Meeting",
        "filing_date": "2026-07-17",
        "accession_number": "1",
        "source_url": "https://nsearchives.nseindia.com/x.pdf",
    }


def test_fetch_company_disclosure_picks_the_most_recent_qualifying_announcement(monkeypatch):
    older = _announcement(seq_id="1", sort_date="2026-04-17 19:07:44", an_dt="17-Apr-2026 19:07:44")
    newer = _announcement(seq_id="2", sort_date="2026-07-17 19:07:44", an_dt="17-Jul-2026 19:07:44")
    # Deliberately out of order -- the client must sort by sort_date
    # itself, not trust the API's own ordering.
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [older, newer])
    monkeypatch.setattr(nse, "_fetch_pdf_bytes", lambda url: b"fake-pdf-bytes")
    monkeypatch.setattr(nse, "_extract_pdf_text", lambda pdf_bytes: "text")

    result = nse.fetch_company_disclosure("RELIANCE")

    assert result["accession_number"] == "2"
    assert result["filing_date"] == "2026-07-17"


def test_fetch_company_disclosure_filters_out_non_results_announcements(monkeypatch):
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [_announcement(desc="Loss of Share Certificates")])

    result = nse.fetch_company_disclosure("RELIANCE")

    assert result is None


def test_fetch_company_disclosure_no_announcements_at_all(monkeypatch):
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [])

    assert nse.fetch_company_disclosure("RELIANCE") is None


def test_fetch_company_disclosure_degrades_to_none_on_network_failure(monkeypatch):
    def _boom(url, params):
        raise ConnectionError("NSE unreachable from this IP")

    monkeypatch.setattr(nse, "_fetch_json", _boom)

    # Must never raise -- same contract as SECEdgarClient.fetch_company_disclosure.
    assert nse.fetch_company_disclosure("RELIANCE") is None


def test_fetch_company_disclosure_degrades_to_none_on_pdf_fetch_failure(monkeypatch):
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [_announcement()])

    def _boom(url):
        raise ConnectionError("nsearchives unreachable")

    monkeypatch.setattr(nse, "_fetch_pdf_bytes", _boom)

    assert nse.fetch_company_disclosure("RELIANCE") is None


def test_fetch_company_disclosure_empty_extracted_text_degrades_to_none(monkeypatch):
    # A scanned filing with no text layer -- out of scope this pass (no OCR).
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [_announcement()])
    monkeypatch.setattr(nse, "_fetch_pdf_bytes", lambda url: b"fake-pdf-bytes")
    monkeypatch.setattr(nse, "_extract_pdf_text", lambda pdf_bytes: "   ")

    assert nse.fetch_company_disclosure("RELIANCE") is None


def test_fetch_company_disclosure_uses_the_cache_and_skips_the_pdf_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [_announcement(seq_id="42")])

    cache_file = tmp_path / "RELIANCE.NS_42.json"
    cached_result = {
        "text": "cached text",
        "form": "Outcome of Board Meeting",
        "filing_date": "2026-01-01",
        "accession_number": "42",
        "source_url": "https://nsearchives.nseindia.com/cached.pdf",
    }
    tmp_path.mkdir(exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_result, f)

    def _boom(url):
        raise AssertionError("_fetch_pdf_bytes should not be called on a cache hit")

    monkeypatch.setattr(nse, "_fetch_pdf_bytes", _boom)

    result = nse.fetch_company_disclosure("RELIANCE")

    assert result == cached_result


def test_nse_filings_client_class_delegates_to_the_module_function(monkeypatch):
    monkeypatch.setattr(nse, "_fetch_json", lambda url, params: [_announcement()])
    monkeypatch.setattr(nse, "_fetch_pdf_bytes", lambda url: b"fake-pdf-bytes")
    monkeypatch.setattr(nse, "_extract_pdf_text", lambda pdf_bytes: "text")

    client = nse.NSEFilingsClient()
    result = client.fetch_company_disclosure("RELIANCE")

    assert result is not None
    assert result["accession_number"] == "1"


def test_parse_announcement_date():
    assert nse._parse_announcement_date("07-Aug-2026 17:07:35") == "2026-08-07"
