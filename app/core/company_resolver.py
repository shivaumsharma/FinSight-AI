"""
company_resolver.py

Detects and resolves company mentions in free text against SEC's real,
comprehensive company/ticker list (~10,000 US-listed companies) --
not a small hardcoded handful. Deterministic (regex extraction + fuzzy
string matching), not LLM-based: entity resolution against a known,
enumerable universe is a lookup problem, and an LLM is an unreliable
way to spell a ticker (same philosophy as sec_edgar_client.py).

How it works
------------
1. Raw ticker tokens ("AAPL", "NFLX") are validated against the real
   ticker universe, not a 5-item list.
2. A small alias table covers companies whose common name doesn't
   resemble their SEC registrant name closely enough for fuzzy
   matching to find on its own (Google -> Alphabet Inc./GOOGL).
3. Capitalized word-runs in the question ("Apple", "NVIDIA's",
   "Bank of America") are extracted as company-name candidates and
   fuzzy-matched against all ~10,000 SEC company titles. A high
   similarity threshold does the work that a hand-built stopword list
   would otherwise need to do -- instruction verbs like "Should" or
   "Analyze" don't closely resemble any real company name.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Optional

import requests
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from app.data.sec_edgar_client import HEADERS

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "filings_cache"
COMPANY_INDEX_CACHE = CACHE_DIR / "company_index.json"
COMPANY_INDEX_TTL_SECONDS = 7 * 24 * 3600

# Companies whose common/colloquial name doesn't closely resemble
# their official SEC registrant name -- fuzzy matching alone would
# miss these.
ALIASES = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "meta": "META",
}

FUZZY_MATCH_THRESHOLD = 85  # rapidfuzz WRatio score, 0-100

# Candidate phrases shorter than this never get fuzzy-matched: short
# strings (e.g. the pronoun "I") can score deceptively high against
# an unrelated long company name under partial-ratio-style scoring,
# since a 1-2 character string trivially "fits inside" almost anything.
MIN_PHRASE_LENGTH = 3

# Sentence-initial instruction verbs, pronouns, and generic words that
# should never be treated as company-name candidates even when
# capitalized (e.g. the "Should"/"I" in "Should I buy Apple?"). A
# defense-in-depth check alongside the fuzzy-match threshold and
# MIN_PHRASE_LENGTH, which already filter most of these out on their
# own.
_STOPWORDS = {
    "i", "should", "analyze", "compare", "generate", "calculate", "teach",
    "what", "how", "why", "please", "give", "tell", "explain", "report",
    "investment", "thesis", "latest", "earnings",
}

_JOINERS = {"of", "and", "the", "&"}

# The LLM-proposal fallback (_llm_propose_company_name) is prone to
# treating a generic finance topic as if it were a company name (e.g.
# proposing "Stock Market" for "what is the stock market", or
# "Finance" for "what is finance") instead of saying NONE as
# instructed -- and terms like these can genuinely fuzzy-match some
# obscure real company's title above the normal threshold ("Stock
# Market" -> London Stock Exchange Group, etc.). Given the project
# must never answer generic finance questions, a false rejection
# (missing a real lowercase-typed company) is far cheaper than a
# false acceptance (running a full analysis on the wrong topic) --
# so proposals matching this denylist are rejected before fuzzy
# matching even runs.
_GENERIC_TOPIC_DENYLIST = {
    "stock market", "stock", "stocks", "market", "markets", "finance",
    "financing", "investing", "investment", "investments", "economy",
    "economics", "trading", "money", "budget", "budgeting", "wall street",
    "retirement", "tax", "taxes", "insurance", "savings", "banking",
    "cryptocurrency", "crypto", "portfolio", "valuation", "dcf",
}

_TICKER_TOKEN = re.compile(r"\b[A-Z]{2,5}\b")
_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z&'’]*")

_index = None


def _fetch_company_index() -> dict:
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()

    tickers = set()
    title_to_ticker = {}

    for entry in raw.values():
        ticker = entry["ticker"].upper()
        title = entry["title"]
        tickers.add(ticker)
        # SEC lists a company's common stock first, then its various
        # preferred-share series under the SAME title afterward (e.g.
        # 17 different tickers all titled "BANK OF AMERICA CORP /DE/").
        # Keeping only the first ticker seen per title keeps the
        # common-stock ticker (BAC) instead of a later entry
        # clobbering it with an obscure preferred series (BML-PJ).
        if title not in title_to_ticker:
            title_to_ticker[title] = ticker

    return {
        "tickers": sorted(tickers),
        "title_to_ticker": title_to_ticker,
    }


def _load_index() -> dict:
    global _index
    if _index is not None:
        return _index

    CACHE_DIR.mkdir(exist_ok=True)

    if COMPANY_INDEX_CACHE.exists():
        age = time.time() - COMPANY_INDEX_CACHE.stat().st_mtime
        if age < COMPANY_INDEX_TTL_SECONDS:
            with open(COMPANY_INDEX_CACHE, "r", encoding="utf-8") as f:
                _index = json.load(f)
                _index["tickers"] = set(_index["tickers"])
                return _index

    data = _fetch_company_index()

    with open(COMPANY_INDEX_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    data["tickers"] = set(data["tickers"])
    _index = data
    return _index


def _extract_candidate_phrases(question: str) -> List[str]:
    """
    Pulls out maximal runs of capitalized words (allowing lowercase
    joiners like "of"/"and"/"the"/"&" mid-run, for names like "Bank
    of America") as company-name candidates.
    """
    tokens = _WORD_TOKEN.findall(question)

    phrases = []
    current = []

    def flush():
        while current and current[-1].lower() in _JOINERS:
            current.pop()
        if current:
            phrases.append(" ".join(current))
        current.clear()

    for tok in tokens:
        # Strip a trailing possessive ("NVIDIA's" -> "NVIDIA") -- .strip()
        # only removes leading/trailing characters and can't do this,
        # since the apostrophe sits mid-token, not at the edge.
        cleaned = re.sub(r"['’]s$", "", tok)
        if not cleaned:
            continue
        lower = cleaned.lower()
        is_cap = cleaned[:1].isupper()

        if is_cap and lower not in _STOPWORDS:
            current.append(cleaned)
        elif lower in _JOINERS and current:
            current.append(cleaned)
        else:
            flush()

    flush()

    return phrases


def _llm_propose_company_name(question: str) -> Optional[str]:
    """
    Fallback for queries with no capitalization to key off at all
    (e.g. "is apple a good stock to buy"). Pure string-similarity
    fallback was tried first and rejected: a single common word like
    "amazon" or "google" can't be reliably told apart from ordinary
    finance vocabulary like "invest" or "earnings" by character
    overlap alone (both score similarly), since real company names
    are often themselves common words.

    This asks the local LLM instead -- but only to name a company, if
    it thinks the question is about one. It never gets to hand back a
    ticker directly; whatever it proposes still has to pass the exact
    same deterministic fuzzy-match validation against SEC's real
    company list as everything else in this module. Worst case it
    proposes nothing, or something that fails validation and
    correctly resolves to no company -- it can't inject a wrong
    ticker into the result on its own.
    """
    from app.core.llm_provider import get_shared_generator

    # Few-shot, matching the pattern already used successfully for the
    # tool planner (see app/planner/llm_planner.py) -- a bare zero-shot
    # version of this prompt was unstable: the model would answer
    # "NONE" and then immediately contradict itself with the right
    # company name on the next line.
    prompt = """Identify the publicly traded company mentioned in a question, if any.

Question: "is apple a good stock to buy right now"
Answer: Apple

Question: "how do i start investing"
Answer: NONE

Question: "should i invest in tesla"
Answer: Tesla

Question: "what is the stock market"
Answer: NONE

Question: "{question}"
Answer:""".format(question=question)

    try:
        raw = get_shared_generator().generate(prompt, max_new_tokens=8)
    except Exception:
        return None

    first_line = raw.strip().splitlines()[0].strip().strip('."\' ')
    # Strip a leading filler word ("Yes Apple" -> "Apple") -- these
    # dilute the fuzzy-match score enough to sometimes tip a tie
    # between the real company and an unrelated same-named one.
    first_line = re.sub(r"^(yes|no|well)[,\s]+", "", first_line, flags=re.IGNORECASE)
    if not first_line or first_line.upper() == "NONE":
        return None
    if first_line.lower() in _GENERIC_TOPIC_DENYLIST:
        return None

    # The model doesn't reliably follow "just a name, or NONE" --
    # CPU inference for this small model isn't even fully
    # deterministic run to run, and it sometimes wraps the real
    # answer in extra chatter ("Yes, Apple has been performing well
    # recently") instead of answering cleanly. Rather than requiring
    # the whole line to be a clean short name, reuse the same
    # capitalized-word-run extractor used on the original question to
    # pull any name-shaped candidate out of whatever it said, and
    # apply the same denylist to each candidate found.
    if len(first_line.split()) <= 4:
        candidates = [first_line]
    else:
        candidates = _extract_candidate_phrases(first_line)

    for candidate in candidates:
        if candidate.lower() not in _GENERIC_TOPIC_DENYLIST and len(candidate.split()) <= 4:
            return candidate

    return None


def _best_title_match(phrase: str, all_titles: List[str]):
    """
    Fuzzy-matches `phrase` against `all_titles`, breaking ties
    properly instead of taking whichever candidate process.extractOne
    happens to return first.

    WRatio has a scoring ceiling (~90.0) that many unrelated titles
    can hit simultaneously for a single common word -- "Apple" scores
    exactly 90.0 against "Apple Inc.", "Apple Hospitality REIT, Inc.",
    "Apple iSports Group, Inc.", and three unrelated pineapple-related
    companies alike, since a short query trivially "fits inside" any
    of them under partial-ratio-style scoring. Taking the first
    match among ties is arbitrary and can silently return the wrong
    company for exactly the well-known, single-word names most
    likely to be asked about. Among candidates tied at the top score,
    this prefers (1) titles that start with the phrase exactly, then
    (2) the shortest title -- both proxies for "the flagship company",
    since obscure/derivative entities tend to have longer, more
    qualified names.
    """
    matches = process.extract(
        phrase, all_titles, scorer=fuzz.WRatio, processor=default_process, limit=25
    )
    if not matches:
        return None

    top_score = matches[0][1]
    tied = [m for m in matches if m[1] >= top_score - 0.01]

    lowered_phrase = phrase.lower()

    def sort_key(m):
        title = m[0]
        starts_with_phrase = not title.lower().startswith(lowered_phrase)  # False sorts first
        return (starts_with_phrase, len(title))

    tied.sort(key=sort_key)
    return tied[0]


def resolve_companies(question: str) -> List[str]:
    """
    Extracts an ordered, de-duplicated list of tickers for companies
    mentioned in `question`. Returns an empty list if none are found
    -- callers should treat that as "no company to research", not
    fall back to a default (there is no default anymore).
    """

    index = _load_index()
    found = []

    for token in _TICKER_TOKEN.findall(question):
        if token in index["tickers"] and token not in found:
            found.append(token)

    lowered = question.lower()
    for alias, ticker in ALIASES.items():
        if alias in lowered and ticker not in found:
            found.append(ticker)

    all_titles = list(index["title_to_ticker"].keys())

    # "and" is ambiguous: it's a legitimate internal word in some
    # company names, but in "Compare Amazon and Google" it's actually
    # separating two different companies, not joining one name. Since
    # the latter is far more common in practice, split any extracted
    # phrase on " and " into independent candidates rather than fuzzy-
    # matching the merged phrase as one (multi-word names like "Bank
    # of America" are unaffected -- they don't contain "and").
    candidate_phrases = []
    for phrase in _extract_candidate_phrases(question):
        if " and " in f" {phrase} ":
            candidate_phrases.extend(part.strip() for part in re.split(r"\s+and\s+", phrase))
        else:
            candidate_phrases.append(phrase)

    # Short (<=5 char) all-uppercase tokens are already checked exactly
    # against the real ticker universe above -- also fuzzy-matching
    # them as company names risks a generic acronym ("DCF", "ROI")
    # coincidentally scoring high against some obscure, unrelated
    # company name.
    def _is_bare_short_acronym(phrase):
        return len(phrase) <= 5 and phrase.isupper() and " " not in phrase

    for phrase in candidate_phrases:
        if phrase.lower() in ALIASES or len(phrase) < MIN_PHRASE_LENGTH:
            continue
        if _is_bare_short_acronym(phrase):
            continue

        match = _best_title_match(phrase, all_titles)
        if match and match[1] >= FUZZY_MATCH_THRESHOLD:
            ticker = index["title_to_ticker"][match[0]]
            if ticker not in found:
                found.append(ticker)

    if not found:
        proposed_name = _llm_propose_company_name(question)
        if proposed_name:
            match = _best_title_match(proposed_name, all_titles)
            if match and match[1] >= FUZZY_MATCH_THRESHOLD:
                ticker = index["title_to_ticker"][match[0]]
                found.append(ticker)

    return found


def is_comparison_question(question: str) -> bool:
    """Heuristic for 'compare X and Y' style questions."""
    keywords = ["compare", " vs ", " vs.", "versus", "better buy", "which is a better"]
    lowered = f" {question.lower()} "
    return any(keyword in lowered for keyword in keywords)
