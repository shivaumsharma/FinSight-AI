"""
Unit tests for select_for_analysis() (app/reporting/news_client.py) --
the capped, category-diverse subset of retrieved news actually fed to
the narrative prompt.

news_client is imported inside each test, not at module level -- its
own module docstring notes it calls load_dotenv() unconditionally at
import time, and a top-level import here would load the real .env
during test collection (before app/tests/test_llm_provider.py's
env-var tests run), leaking real LLM_BASE_URL/LLM_API_KEY/LLM_MODEL
values into what those tests expect to be a clean environment. Matches
the same deferred-import pattern already used for this in
app/tests/test_llm_provider.py and app/tests/test_narrative_builder.py.
"""


def _article(headline, date, categories=("other",), url=None):
    return {
        "headline": headline,
        "source": "Test Source",
        "date": date,
        "url": url or headline,
        "summary": "",
        "categories": list(categories),
    }


def test_select_for_analysis_prefers_company_specific_over_generic_same_day_noise():
    """
    Reproduces the reported bug on a real MSFT run: generic same-day
    market-roundup stories that only namechecked the ticker in
    Finnhub's feed crowded out older, genuinely Microsoft-specific
    analysis. Company-specific coverage must be exhausted first,
    regardless of recency.
    """
    from app.reporting.news_client import MAX_SELECTED, select_for_analysis

    generic_recent = [
        _article("Dow Rises, Nasdaq Sinks As Chipmakers Dive; Coca-Cola Surges", "2026-07-28"),
        _article("Buy or Sell the Memory Plunge? SanDisk Sinks 9%, Micron Drops 7%", "2026-07-28"),
        _article("Micron Stock Falls After an Otherwise Strong Week", "2026-07-28"),
        _article("Meta, BlackRock Plan to Invest in $14 Billion Data Center", "2026-07-28"),
        _article("This Week In Cloud AI -- Axe Compute Secures $1.5 Billion Deal", "2026-07-28"),
    ]
    specific_older = [
        _article(
            "Microsoft Plans $190 Billion of Capital Spending This Year. "
            "Wednesday Shows Whether Azure Is Keeping Up.", "2026-07-24",
        ),
        _article("Microsoft faces AI capex scrutiny as it prepares to report Q4 results", "2026-07-23"),
        _article("Microsoft: The Market Is Right To Discount The $190 Billion Bet", "2026-07-22"),
        _article("Guggenheim Reiterates Buy on Microsoft, Maintains $586 Price Target", "2026-07-21"),
        _article("Morgan Stanley resets Microsoft stock forecast ahead of earnings", "2026-07-20"),
        _article("Microsoft Stock Is Near a 1-Year Low", "2026-07-19"),
        _article("Does Azure AI Growth Make Microsoft Stock a Buy Ahead of Q4 Earnings?", "2026-07-18"),
        _article("Microsoft Earnings Due July 29 as Options Signal 7% Stock Move", "2026-07-17"),
    ]
    # Already sorted most-recent-first, matching fetch_company_news's output order.
    articles = generic_recent + specific_older
    assert len(articles) > MAX_SELECTED

    selected = select_for_analysis(articles, ticker="MSFT", company_name="Microsoft Corporation")
    selected_headlines = {a["headline"] for a in selected}

    for a in specific_older:
        assert a["headline"] in selected_headlines, f"missing company-specific article: {a['headline']}"

    generic_selected = [h for h in selected_headlines if any(h == g["headline"] for g in generic_recent)]
    # 8 specific articles exist and MAX_SELECTED is 10 -- exactly 2 generic
    # slots should remain, filled by recency among the generic pool.
    assert len(generic_selected) == 2
    assert "Dow Rises, Nasdaq Sinks As Chipmakers Dive; Coca-Cola Surges" in generic_selected
    assert "Buy or Sell the Memory Plunge? SanDisk Sinks 9%, Micron Drops 7%" in generic_selected


def test_select_for_analysis_falls_back_to_generic_when_no_specific_coverage_in_category():
    from app.reporting.news_client import select_for_analysis

    articles = [
        _article(f"Generic litigation-adjacent story {i}", f"2026-07-{20 + i:02d}", categories=("litigation",))
        for i in range(12)
    ]
    # No headline mentions the company at all -- nothing is "specific",
    # so category-diversity + recency fallback should still populate
    # the litigation slots from the generic pool rather than coming up empty.
    selected = select_for_analysis(articles, ticker="ZZZZ", company_name="Zzz Corp")
    assert len(selected) == 10
    assert all("litigation" in a["categories"] for a in selected)


def test_select_for_analysis_degrades_to_ticker_only_matching_without_company_name():
    from app.reporting.news_client import select_for_analysis

    articles = [_article(f"MSFT-tagged wire story {i}", f"2026-07-{10 + i:02d}") for i in range(12)]
    articles[0]["headline"] = "MSFT rallies on strong cloud demand"
    # company_name unavailable (e.g. market_data_tool didn't run for
    # this plan) -- must not raise, and should still surface the one
    # headline that names the ticker directly.
    selected = select_for_analysis(articles, ticker="MSFT", company_name=None)
    assert len(selected) == 10
    assert "MSFT rallies on strong cloud demand" in {a["headline"] for a in selected}


def test_select_for_analysis_returns_everything_under_the_cap():
    from app.reporting.news_client import select_for_analysis

    articles = [_article(f"Story {i}", "2026-07-01") for i in range(5)]
    assert select_for_analysis(articles, ticker="MSFT", company_name="Microsoft") == articles


def test_centrality_score_boosts_material_event_headlines_over_incidental_mentions():
    """
    Mention-count alone can't separate "the story that explains the
    quarter" from an incidental namecheck -- confirmed on a real MSFT
    run where "IBM (IBM) Joins Nvidia And Microsoft In Open Secure AI
    Alliance" (one mention, no commas) tied "Microsoft Plans $190
    Billion of Capital Spending" (also one mention, no commas) and won
    the recency tiebreak, crowding out the capex story that actually
    explained the report's own DCF/FCF gap. _materiality_bonus() must
    give the capex/earnings headline a strictly higher score.
    """
    from app.reporting.news_client import _centrality_score, _primary_company_term

    company_term = _primary_company_term("Microsoft Corporation")
    incidental = _article(
        "IBM (IBM) Joins Nvidia And Microsoft In Open Secure AI Alliance", "2026-07-27",
    )
    material = _article(
        "Microsoft Plans $190 Billion of Capital Spending This Year", "2026-07-27",
    )
    assert (
        _centrality_score(material, "MSFT", company_term)
        > _centrality_score(incidental, "MSFT", company_term)
    )


def test_select_for_analysis_ranks_dedicated_piece_over_same_day_multi_company_listicle():
    """
    Binary specific/generic membership isn't enough on its own: a
    same-day multi-company listicle that merely namechecks the ticker
    still lands in the "specific" pool and would win a pure-recency
    tiebreak against an older, genuinely dedicated piece. Centrality
    scoring (mentions minus a comma-count listicle penalty) has to
    settle that tie the other way.
    """
    from app.reporting.news_client import select_for_analysis

    listicle = _article(
        "Alphabet, Microsoft, Meta Platforms, Apple and Amazon are part of Zacks Earnings Preview",
        "2026-07-27",
    )
    multi_company = _article("Nvidia, Microsoft, Palantir Lead Big Tech Revolt", "2026-07-27")
    dedicated_piece = _article(
        "Microsoft Plans $190 Billion of Capital Spending This Year. "
        "Wednesday Shows Whether Azure Is Keeping Up.",
        "2026-07-24",
    )
    filler = [_article(f"Unrelated market story {i}", "2026-07-01") for i in range(9)]

    articles = [listicle, multi_company] + filler + [dedicated_piece]
    assert len(articles) > 10

    selected_headlines = [
        a["headline"]
        for a in select_for_analysis(articles, ticker="MSFT", company_name="Microsoft Corporation")
    ]

    assert dedicated_piece["headline"] in selected_headlines
    dedicated_idx = selected_headlines.index(dedicated_piece["headline"])
    # Both distractors are three days newer than the dedicated piece --
    # if recency were still deciding this, they'd sort ahead of it.
    for distractor in (listicle, multi_company):
        if distractor["headline"] in selected_headlines:
            assert dedicated_idx < selected_headlines.index(distractor["headline"]), (
                f"{distractor['headline']!r} outranked the dedicated analysis piece"
            )


def test_primary_company_term_skips_generic_leading_words():
    from app.reporting.news_client import _primary_company_term

    assert _primary_company_term("Microsoft Corporation") == "Microsoft"
    assert _primary_company_term("The Walt Disney Company") not in ("The", "Company")
    assert _primary_company_term("International Business Machines") not in ("International", "Business")
    assert _primary_company_term("First Solar, Inc.") not in ("First", "Inc")
    assert _primary_company_term(None) is None
    assert _primary_company_term("") is None


def test_ticker_matching_requires_word_boundaries_and_uppercase():
    from app.reporting.news_client import _is_company_specific

    # "ALL" (Allstate) must not match as a substring of "ALLOCATION" or
    # the lowercase word "all" -- only a standalone, uppercase "ALL".
    assert _is_company_specific(_article("ALL climbs after earnings beat", "2026-07-20"), "ALL", None) is True
    assert _is_company_specific(_article("Fund allocation strategy shifts again", "2026-07-20"), "ALL", None) is False
    assert _is_company_specific(_article("Investors weigh it all before earnings", "2026-07-20"), "ALL", None) is False

    # "IT" (Gartner) must not match the pronoun "it"/"It".
    assert _is_company_specific(_article("It was a quiet day for markets", "2026-07-20"), "IT", None) is False
    assert _is_company_specific(_article("IT spending outlook raised for 2026", "2026-07-20"), "IT", None) is True
