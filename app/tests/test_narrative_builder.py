"""
Unit tests for app/reporting/narrative_builder.py: _split_sections()
(the heading-based parser that turns one combined LLM narrative call
into {section_name: text}) and the prompt-construction helpers that
force the model to engage with specific facts -- FCF-vs-revenue growth
divergence and an imminent earnings date -- rather than letting them
get lost among everything else in the prompt.

narrative_builder is imported inside each test, not at module level --
it pulls in app.reporting.news_client, whose module docstring notes it
calls load_dotenv() unconditionally at import time. A top-level import
here would load the real .env during test collection (before any test,
including app/tests/test_llm_provider.py's env-var tests, has run),
leaking real LLM_BASE_URL/LLM_API_KEY/LLM_MODEL values into what those
tests expect to be a clean environment. Matches the same deferred-import
pattern already used for this in app/tests/test_llm_provider.py.
"""

import logging
from datetime import date, timedelta


def test_split_sections_basic_two_section_split():
    from app.reporting.narrative_builder import _split_sections

    raw = (
        "# Executive Summary\n"
        "The company shows solid fundamentals.\n\n"
        "# Business Analysis\n"
        "The business operates in a competitive market.\n"
    )
    sections = _split_sections(raw)
    assert sections["Executive Summary"] == "The company shows solid fundamentals."
    assert sections["Business Analysis"] == "The business operates in a competitive market."


def test_split_sections_duplicate_heading_keeps_first_complete_section(caplog):
    from app.reporting.narrative_builder import _split_sections

    # Simulates model drift: a complete "Executive Summary" is written,
    # then the model loops back and re-emits the same heading with a
    # truncated, incomplete restart before eventually moving on.
    raw = (
        "# Executive Summary\n"
        "This is the complete executive summary covering valuation, "
        "growth, and risk in full detail.\n\n"
        "# Executive Summary\n"
        "This is an incomplete restart that got cut\n\n"
        "# Business Analysis\n"
        "The business operates in a competitive market.\n"
    )

    with caplog.at_level(logging.WARNING):
        sections = _split_sections(raw)

    assert sections["Executive Summary"] == (
        "This is the complete executive summary covering valuation, "
        "growth, and risk in full detail."
    )
    assert "incomplete restart" not in sections["Executive Summary"]
    assert sections["Business Analysis"] == "The business operates in a competitive market."

    assert any(
        "duplicate heading" in record.message and "Executive Summary" in record.message
        for record in caplog.records
    )
    assert all(record.levelno == logging.WARNING for record in caplog.records)


def test_split_sections_no_duplicate_heading_logs_nothing(caplog):
    from app.reporting.narrative_builder import _split_sections

    raw = "# Executive Summary\nJust one occurrence.\n"
    with caplog.at_level(logging.WARNING):
        _split_sections(raw)
    assert caplog.records == []


# ---------------------------------------------------------------- _growth_divergence_block

def test_growth_divergence_block_fires_when_fcf_declines_while_revenue_grows():
    """
    Reproduces the real MSFT case: Revenue +14.93%, Net Income +15.54%,
    FCF -3.32% -- the report's own DCF-vs-relative-valuation gap, which
    the narrative previously never mentioned at all.
    """
    from app.reporting.narrative_builder import _growth_divergence_block

    growth = {
        "Revenue Growth (%)": 14.93, "Net Income Growth (%)": 15.54,
        "FCF Growth (%)": -3.32, "FCF Trend": "Declining",
    }
    block = _growth_divergence_block(growth)
    assert "GROWTH DIVERGENCE" in block
    assert "-3.32" in block


def test_growth_divergence_block_silent_when_fcf_growth_is_positive():
    from app.reporting.narrative_builder import _growth_divergence_block

    growth = {"Revenue Growth (%)": 10.0, "Net Income Growth (%)": 8.0, "FCF Growth (%)": 5.0}
    assert _growth_divergence_block(growth) == ""


def test_growth_divergence_block_silent_when_everything_is_declining_together():
    # FCF declining isn't a "divergence" worth flagging if revenue and
    # earnings are declining right along with it -- that's just a
    # company doing badly across the board, not a DCF-specific gap.
    from app.reporting.narrative_builder import _growth_divergence_block

    growth = {"Revenue Growth (%)": -5.0, "Net Income Growth (%)": -8.0, "FCF Growth (%)": -3.32}
    assert _growth_divergence_block(growth) == ""


def test_growth_divergence_block_silent_when_fcf_growth_unavailable():
    from app.reporting.narrative_builder import _growth_divergence_block

    growth = {"Revenue Growth (%)": 10.0, "Net Income Growth (%)": 8.0, "FCF Growth (%)": "Unavailable"}
    assert _growth_divergence_block(growth) == ""


# ---------------------------------------------------------------- _earnings_proximity_block

def test_earnings_proximity_block_fires_within_the_window():
    from app.reporting.narrative_builder import _earnings_proximity_block

    block = _earnings_proximity_block(date.today() + timedelta(days=1))
    assert "UPCOMING EARNINGS" in block


def test_earnings_proximity_block_fires_for_today():
    from app.reporting.narrative_builder import _earnings_proximity_block

    block = _earnings_proximity_block(date.today())
    assert "UPCOMING EARNINGS" in block
    assert "today" in block


def test_earnings_proximity_block_silent_when_none():
    from app.reporting.narrative_builder import _earnings_proximity_block

    assert _earnings_proximity_block(None) == ""


def test_earnings_proximity_block_silent_when_far_out():
    from app.reporting.narrative_builder import EARNINGS_PROXIMITY_DAYS, _earnings_proximity_block

    far = date.today() + timedelta(days=EARNINGS_PROXIMITY_DAYS + 10)
    assert _earnings_proximity_block(far) == ""


# ---------------------------------------------------------------- _build_prompt wiring

def test_build_prompt_includes_both_blocks_when_applicable():
    from app.reporting.narrative_builder import _build_prompt

    class _FakeContext:
        ticker = "MSFT"
        news_selected = []
        news_articles = []
        research_summary = ""

    report_data = {
        "ticker": "MSFT",
        "company_overview": {
            "name": "Microsoft Corporation", "sector": "Technology", "industry": "Software",
            "business_summary": "Microsoft develops software and cloud services.",
        },
        "growth_analysis": {
            "Revenue Growth (%)": 14.93, "Revenue Trend": "Healthy Growth",
            "Net Income Growth (%)": 15.54, "FCF Growth (%)": -3.32, "FCF Trend": "Declining",
        },
        "valuation_analysis": {
            "Intrinsic Value (per share)": 340.81, "Current Price": 393.35, "Upside (%)": -13.4,
            "WACC": 0.1062, "Terminal Growth Rate": 0.04,
        },
        "market_earnings_snapshot": {
            "current_price": 393.35, "sentiment_label": "Positive", "sentiment_confidence": "74.83%",
            "news_sentiment_label": "Neutral", "news_sentiment_confidence": "60%",
            "next_earnings_date": date.today() + timedelta(days=1),
        },
        "recommendation": {"rating": "Hold", "basis": "test basis"},
    }

    prompt = _build_prompt(_FakeContext(), report_data)

    # Colon distinguishes the actual data block from the static Rules
    # line ("If a GROWTH DIVERGENCE or UPCOMING EARNINGS block appears
    # above...") that mentions both terms unconditionally.
    assert "GROWTH DIVERGENCE:" in prompt
    assert "UPCOMING EARNINGS:" in prompt
    assert "10.62%" in prompt  # WACC formatted as a percent, not the raw 0.1062 fraction
    assert "4.00%" in prompt


def test_build_prompt_omits_both_blocks_when_not_applicable():
    from app.reporting.narrative_builder import _build_prompt

    class _FakeContext:
        ticker = "AAPL"
        news_selected = []
        news_articles = []
        research_summary = ""

    report_data = {
        "ticker": "AAPL",
        "company_overview": {
            "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
            "business_summary": "Apple designs and sells consumer electronics.",
        },
        "growth_analysis": {
            "Revenue Growth (%)": 5.0, "Revenue Trend": "Stable",
            "Net Income Growth (%)": 4.0, "FCF Growth (%)": 3.0, "FCF Trend": "Stable",
        },
        "valuation_analysis": {
            "Intrinsic Value (per share)": 150.0, "Current Price": 170.0, "Upside (%)": -11.8,
            "WACC": 0.09, "Terminal Growth Rate": 0.03,
        },
        "market_earnings_snapshot": {
            "current_price": 170.0, "sentiment_label": "Neutral", "sentiment_confidence": "60%",
            "news_sentiment_label": "Neutral", "news_sentiment_confidence": "60%",
            "next_earnings_date": None,
        },
        "recommendation": {"rating": "Hold", "basis": "test basis"},
    }

    prompt = _build_prompt(_FakeContext(), report_data)

    assert "GROWTH DIVERGENCE:" not in prompt
    assert "UPCOMING EARNINGS:" not in prompt
