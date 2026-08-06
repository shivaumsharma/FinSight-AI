"""
news_client.py

Fetches recent company-specific news via Finnhub's free company-news
endpoint, for the news-grounded Risk Analysis and Market/Media
Sentiment features.

Requires a free Finnhub API key (https://finnhub.io) set as
FINNHUB_API_KEY in a local .env file (never committed -- see
.env.example), loaded via python-dotenv. Missing key, network
failure, or a ticker with no recent coverage all degrade to an empty
list rather than raising -- callers should treat that as
"insufficient news coverage", not an error (see report_data_builder.py
and narrative_builder.py, which are expected to say so explicitly
rather than invent generic filler when this returns []).

Categorization is deterministic (keyword matching), not an LLM -- so
a "no news found in this risk category" claim in the report is
something the code actually checked, not something the model guessed.
"""

import os
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "filings_cache"
NEWS_CACHE_TTL_SECONDS = 24 * 3600  # respect Finnhub's free-tier rate limit

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

DAYS_LOOKBACK = 60
MAX_SELECTED = 10
PER_CATEGORY_MIN = 2

# Deliberately simple keyword sets for v1 -- an article can match more
# than one category. Miscategorization/missed nuance is an accepted
# tradeoff for a fully explainable, non-LLM classification step.
RISK_CATEGORIES = ("litigation", "regulatory", "competitive", "macro")

_RISK_CATEGORY_KEYWORDS = {
    "litigation": ["lawsuit", "litigation", "sue", "sued", "settlement", "court", "legal action"],
    "regulatory": ["regulator", "regulation", "antitrust", "compliance", "fine", "investigation", "ftc", "sec ", "doj"],
    "competitive": ["competitor", "competition", "market share", "rival"],
    "macro": ["inflation", "interest rate", "recession", "tariff", "fed ", "federal reserve", "economic"],
}

# Same "deliberately simple, explainable keyword set" tradeoff as
# _RISK_CATEGORY_KEYWORDS above, used by _centrality_score() to boost
# headlines about a genuine material financial event over ones that
# merely namecheck the company. Mention-count alone can't tell "the
# story explaining the quarter" from incidental mentions ("IBM (IBM)
# Joins Nvidia And Microsoft In Open Secure AI Alliance" scored the
# same as "Microsoft Plans $190 Billion of Capital Spending" on
# mentions alone) -- confirmed on a real MSFT run where several
# incidental-mention headlines outranked the capex story that actually
# explained the report's own DCF/FCF gap.
_MATERIALITY_KEYWORDS = (
    "capex", "capital spending", "capital expenditure",
    "earnings", "guidance", "quarterly results",
    "beat", "misses", "acquisition", "merger", "buyback", "dividend",
    "layoffs", "restructuring",
)


def _categorize(headline: str, summary: str) -> List[str]:
    text = f"{headline} {summary}".lower()
    matched = [
        category
        for category, keywords in _RISK_CATEGORY_KEYWORDS.items()
        if any(kw in text for kw in keywords)
    ]
    return matched or ["other"]


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"news_{ticker.upper()}.json"


# Separate cache/TTL from company news above -- general market
# headlines are worth refreshing far more often than the 24h TTL that
# makes sense for a single ticker's coverage (a report is generated
# once; the home dashboard's news feed gets checked repeatedly through
# the day), and there's exactly one of these regardless of how many
# tickers a user follows, so a single shared cache file is enough.
MARKET_NEWS_CACHE_FILE = CACHE_DIR / "market_news.json"
MARKET_NEWS_CACHE_TTL_SECONDS = 30 * 60


def fetch_market_news(limit: int = 20) -> List[Dict]:
    """
    General market news (not tied to any one company) via Finnhub's
    /news?category=general endpoint -- a list of {"headline", "source",
    "date" (YYYY-MM-DD), "url", "summary"} dicts, most recent first.
    Same never-raise contract as fetch_company_news: a missing API key,
    a failed request, or an unexpected response shape all degrade to
    an empty list rather than propagating an error to the caller.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    if MARKET_NEWS_CACHE_FILE.exists():
        age = time.time() - MARKET_NEWS_CACHE_FILE.stat().st_mtime
        if age < MARKET_NEWS_CACHE_TTL_SECONDS:
            with open(MARKET_NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)[:limit]

    if not FINNHUB_API_KEY:
        return []

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": FINNHUB_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        raw_articles = resp.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(raw_articles, list):
        return []

    # Sorted by the raw numeric timestamp (not the formatted date
    # string) before formatting -- Finnhub's own response order isn't
    # guaranteed to be strictly chronological.
    raw_articles.sort(key=lambda item: item.get("datetime") or 0, reverse=True)

    articles = []
    seen_urls = set()
    for item in raw_articles:
        headline = item.get("headline")
        url = item.get("url")
        if not headline or not url or url in seen_urls:
            continue
        seen_urls.add(url)

        date_str = (
            datetime.utcfromtimestamp(item["datetime"]).strftime("%Y-%m-%d")
            if item.get("datetime")
            else "date unknown"
        )

        articles.append({
            "headline": headline,
            "source": item.get("source", "Unknown"),
            "date": date_str,
            "url": url,
            "summary": item.get("summary", ""),
        })

    with open(MARKET_NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f)

    return articles[:limit]


def fetch_company_news(ticker: str, days: int = DAYS_LOOKBACK) -> List[Dict]:
    """
    Returns a list of {"headline", "source", "date" (YYYY-MM-DD),
    "url", "summary", "categories"} for `ticker`'s last `days` of
    news, most recent first. Returns [] if FINNHUB_API_KEY isn't set,
    the request fails, or there's no coverage -- never raises.
    """

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _cache_path(ticker)

    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < NEWS_CACHE_TTL_SECONDS:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

    if not FINNHUB_API_KEY:
        return []

    try:
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=days)

        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker.upper(),
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": FINNHUB_API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        raw_articles = resp.json()
    except (requests.RequestException, ValueError):
        return []

    if not isinstance(raw_articles, list):
        return []

    articles = []
    for item in raw_articles:
        headline = item.get("headline")
        url = item.get("url")
        if not headline or not url:
            continue

        summary = item.get("summary", "")
        date_str = (
            datetime.utcfromtimestamp(item["datetime"]).strftime("%Y-%m-%d")
            if item.get("datetime")
            else "date unknown"
        )

        articles.append({
            "headline": headline,
            "source": item.get("source", "Unknown"),
            "date": date_str,
            "url": url,
            "summary": summary,
            "categories": _categorize(headline, summary),
        })

    articles.sort(key=lambda a: a["date"], reverse=True)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(articles, f)

    return articles


# Skipped when picking a single distinctive term out of a company's
# legal name (see _primary_company_term): leading articles, corporate
# entity-type filler, and generic business words that would otherwise
# match nearly every headline (e.g. "The Walt Disney Company" -> "The"
# is wrong the same way "International Business Machines" -> either
# "International" or "Business" is wrong -- these are the words most
# likely to appear in an unrelated headline, not the words a reader
# would recognize the company by).
_GENERIC_NAME_WORDS = {
    "the", "a", "an", "and", "of",
    "inc", "incorporated", "corp", "corporation", "company", "companies",
    "co", "ltd", "limited", "plc", "group", "holding", "holdings",
    "international", "first", "general", "national", "global", "united",
    "american", "business", "industries", "systems", "technology",
    "technologies", "enterprises", "solutions", "worldwide",
}

_NAME_WORD_PATTERN = re.compile(r"[A-Za-z']+")


def _primary_company_term(company_name: Optional[str]) -> Optional[str]:
    """
    Picks one distinctive word out of the company's legal name (e.g.
    "Microsoft" from "Microsoft Corporation") to match against
    headlines. Naively taking the first word breaks on real company
    names -- "The Walt Disney Company", "International Business
    Machines", "First Solar" would yield "The"/"International"/"First",
    each a generic word that matches nearly every headline in
    existence, not a signal of anything. Skips _GENERIC_NAME_WORDS and
    any word under 4 characters; returns None (callers fall back to
    ticker-only matching) if nothing usable remains.
    """
    if not company_name:
        return None
    for word in _NAME_WORD_PATTERN.findall(company_name):
        if len(word) < 4 or word.lower() in _GENERIC_NAME_WORDS:
            continue
        return word
    return None


def _term_occurrences(headline: str, term: Optional[str], case_sensitive: bool) -> int:
    """
    Word-boundary count, not substring count -- a plain `in` check
    would match ticker "ALL" (Allstate) inside "ALLOCATION", "IT"
    (Gartner) inside nearly any sentence containing the pronoun "it",
    "ON" (ON Semiconductor) inside "MONDAY", and so on for every short
    ticker. Ticker matching stays case-SENSITIVE deliberately (a
    literal uppercase "ALL"/"IT"/"ON" as its own word is a real ticker
    mention; the same letters lowercase are just English words) --
    company-name-word matching stays case-insensitive since headlines
    capitalize proper nouns naturally but inconsistently.
    """
    if not term:
        return 0
    flags = 0 if case_sensitive else re.IGNORECASE
    return len(re.findall(rf"\b{re.escape(term)}\b", headline, flags))


def _mention_count(article: Dict, ticker: str, company_term: Optional[str]) -> int:
    headline = article.get("headline", "")
    return (
        _term_occurrences(headline, ticker, case_sensitive=True)
        + _term_occurrences(headline, company_term, case_sensitive=False)
    )


def _materiality_bonus(headline: str) -> int:
    lowered = headline.lower()
    return sum(1 for kw in _MATERIALITY_KEYWORDS if kw in lowered)


def _is_company_specific(article: Dict, ticker: str, company_term: Optional[str]) -> bool:
    """
    True if the headline is actually ABOUT this company, not just
    something Finnhub's per-ticker company-news feed loosely
    associated with it (a same-sector peer's earnings, a broad
    market-index roundup that namechecks a dozen tickers, an
    unrelated company's partnership announcement). Headline-only by
    design -- a piece that's genuinely about the company but whose
    headline doesn't name it (a cute/oblique title like "The $190
    Billion Question") scores zero here; body-text matching was
    considered out of scope for this pass.
    """
    return _mention_count(article, ticker, company_term) > 0


def _centrality_score(article: Dict, ticker: str, company_term: Optional[str]) -> int:
    """
    Ranks WITHIN the company-specific pool -- binary specific/generic
    membership alone still let a same-day multi-company listicle that
    namechecks this ticker in passing (e.g. a 5-company earnings-
    preview roundup) win a recency tiebreak against an older, dedicated
    analysis piece, which is exactly backwards: confirmed on a real
    run where a Zacks multi-company preview and a "Nvidia, Microsoft,
    Palantir Lead Big Tech Revolt" story would have outranked a
    genuinely Microsoft-specific capex analysis on recency alone.
    Financial headlines listing several companies are reliably comma-
    separated ("Alphabet, Microsoft, Meta Platforms, Apple and Amazon
    are part of..."); a headline actually about one company rarely has
    more than one comma, and when it does (e.g. "Guggenheim Reiterates
    Buy on Microsoft, Maintains $586 Price Target"), the mention count
    still keeps the score positive. Score = how many times this
    company is actually named, minus a comma-count proxy for "how many
    other companies this headline is also about", plus a
    _materiality_bonus() for headlines about an actual financial event
    (capex, earnings, guidance, an acquisition, ...) rather than an
    incidental namecheck -- mention-count alone can't tell "Microsoft
    Plans $190 Billion of Capital Spending" (the story that explains
    the report's own DCF/FCF gap) apart from "IBM (IBM) Joins Nvidia
    And Microsoft In Open Secure AI Alliance" (also one mention, no
    commas, but not the story anyone actually needs).
    """
    headline = article.get("headline", "")
    return (
        _mention_count(article, ticker, company_term)
        - headline.count(",")
        + _materiality_bonus(headline)
    )


def select_for_analysis(
    articles: List[Dict],
    ticker: str = "",
    company_name: Optional[str] = None,
    max_count: int = MAX_SELECTED,
) -> List[Dict]:
    """
    Picks a capped subset for the LLM prompt: company-specific
    relevance first (ranked by _centrality_score within that pool),
    then category diversity (so Risk Analysis has real material for
    each risk type that has any coverage), then recency to break any
    remaining ties. The full, unfiltered list from fetch_company_news
    stays available separately for the transparency panel -- callers
    must not treat this selection as "everything that was retrieved."

    Earlier versions ranked purely by recency (inherited from
    fetch_company_news's sort order) within each category and in the
    final fill pass -- confirmed on a real run that this lets a
    same-day broad-market story that only namechecks the ticker in
    passing (a sector-wide selloff, a Dow/Nasdaq wrap, a multi-company
    listicle) crowd out older, genuinely company-specific analysis,
    which is exactly backwards for a report meant to explain THIS
    company. `ticker`/`company_name` (pass company_name="" or leave it
    None if unavailable -- e.g. market_data_tool didn't run for this
    plan -- and this falls back to ticker-only matching) split articles
    into "actually about this company" vs. "merely associated with it"
    via `_is_company_specific()`; the specific pool -- sorted highest-
    centrality-first -- is exhausted per category, then overall, before
    the generic pool (still recency-ordered) is used at all.
    """
    if len(articles) <= max_count:
        return list(articles)

    company_term = _primary_company_term(company_name)

    # sorted() is stable, so among equal centrality scores this
    # preserves `articles`'s incoming recency order -- recency still
    # breaks ties, it just no longer outranks relevance.
    specific = sorted(
        (a for a in articles if _is_company_specific(a, ticker, company_term)),
        key=lambda a: _centrality_score(a, ticker, company_term),
        reverse=True,
    )
    specific_urls = {a["url"] for a in specific}
    generic = [a for a in articles if a["url"] not in specific_urls]

    selected = []
    selected_urls = set()

    def take(pool, limit=None):
        count = 0
        for a in pool:
            if len(selected) >= max_count or (limit is not None and count >= limit):
                return
            if a["url"] in selected_urls:
                continue
            selected.append(a)
            selected_urls.add(a["url"])
            count += 1

    for category in RISK_CATEGORIES:
        before = len(selected)
        take([a for a in specific if category in a["categories"]], PER_CATEGORY_MIN)
        filled = len(selected) - before
        if filled < PER_CATEGORY_MIN:
            take([a for a in generic if category in a["categories"]], PER_CATEGORY_MIN - filled)

    take(specific)
    take(generic)

    return selected[:max_count]
