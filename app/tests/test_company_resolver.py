"""
Unit tests for company_resolver.py.

Deliberately not mocked: this module's whole design point (see its own
module docstring) is resolving against the real, live SEC and NSE
company universes, not a small hardcoded list -- a mocked test would
validate the mock, not the actual matching behavior that produced the
"bajaj finance" -> Roma Green Finance Limited bug this file exists to
catch a regression of. Needs network access (SEC/NSE index fetch, both
cached after the first call) and a configured LLM provider (used by
the lowercase-query fallback path).

resolve_companies is imported inside each test, not at module level --
company_resolver pulls in app.core.llm_provider indirectly, and
loading .env at module import time (during test collection, before
app/tests/test_llm_provider.py's clean-env tests run) would leak real
LLM_BASE_URL/LLM_API_KEY/LLM_MODEL into what those tests expect to be
a clean environment. Matches the same deferred-import pattern already
used for this in app/tests/test_llm_provider.py and
app/tests/test_narrative_builder.py (see their own docstrings/comments).
"""


def _resolve(question: str):
    from dotenv import load_dotenv

    load_dotenv()
    from app.core.company_resolver import resolve_companies

    return resolve_companies(question)


def test_lowercase_indian_company_resolves_to_nse_ticker():
    # The exact bug this file guards against: an all-lowercase mention
    # of a real NSE-listed company used to silently resolve to an
    # unrelated US micro-cap (Roma Green Finance Limited) that merely
    # shared the word "Finance" with the LLM-proposed name, because
    # the resolver only ever fuzzy-matched against SEC's US universe.
    assert _resolve("should i buy bajaj finance") == ["BAJFINANCE.NS"]


def test_lowercase_us_company_still_resolves_correctly():
    assert _resolve("is apple a good stock to buy right now") == ["AAPL"]


def test_lowercase_us_company_tesla_still_resolves_correctly():
    assert _resolve("should i invest in tesla") == ["TSLA"]


def test_raw_ticker_token_still_resolves():
    assert _resolve("what do you think about AAPL") == ["AAPL"]


def test_compare_question_splits_on_and():
    result = _resolve("Compare Amazon and Google")
    assert set(result) == {"AMZN", "GOOGL"}


def test_generic_finance_question_resolves_to_nothing():
    assert _resolve("what is the best way to start investing") == []


def test_explicit_nse_suffixed_ticker_resolves_directly():
    # A bare "SYMBOL.NS" written straight into free text (e.g. a
    # research-trigger link built from a Portfolio holding's own
    # ticker) used to resolve to nothing: _TICKER_TOKEN caps at 5
    # letters (many real NSE symbols run longer) and even a match would
    # only have been checked against the US ticker set, never NSE's.
    assert _resolve("Should I invest in BAJFINANCE.NS?") == ["BAJFINANCE.NS"]


def test_explicit_nse_suffixed_ticker_is_case_insensitive():
    assert _resolve("what about bajfinance.ns") == ["BAJFINANCE.NS"]


def test_explicit_nse_suffixed_ticker_combines_with_a_us_ticker():
    result = _resolve("Compare AAPL and BAJFINANCE.NS")
    assert set(result) == {"AAPL", "BAJFINANCE.NS"}


def test_similar_nse_company_names_disambiguate_correctly():
    # Multiple real "Bajaj ..." companies exist on NSE -- exercises
    # that the fuzzy matcher (and its start-with-phrase/shortest-title
    # tie-break) picks the right one out of several genuinely similar
    # real names, not just "any Bajaj company".
    assert _resolve("should i buy bajaj auto") == ["BAJAJ-AUTO.NS"]
    assert _resolve("is bajaj finserv a good buy") == ["BAJAJFINSV.NS"]


# ---------------------------------------------------------------- suggest_companies

def _suggest(prefix: str, limit: int = 8):
    from dotenv import load_dotenv

    load_dotenv()
    from app.core.company_resolver import suggest_companies

    return suggest_companies(prefix, limit=limit)


def test_suggest_companies_matches_a_ticker_prefix():
    results = _suggest("AAP")
    assert any(r["ticker"] == "AAPL" for r in results)
    assert all("ticker" in r and "name" in r for r in results)


def test_suggest_companies_matches_a_real_company_by_name_prefix():
    results = _suggest("Tata Steel")
    assert any(r["ticker"] == "TATASTEEL.NS" for r in results)


def test_suggest_companies_respects_the_limit():
    results = _suggest("A", limit=3)
    assert len(results) <= 3


def test_suggest_companies_empty_prefix_returns_nothing():
    assert _suggest("") == []


def test_suggest_companies_never_returns_duplicate_tickers():
    results = _suggest("Tata")
    tickers = [r["ticker"] for r in results]
    assert len(tickers) == len(set(tickers))
