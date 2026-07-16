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
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

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


def select_for_analysis(articles: List[Dict], max_count: int = MAX_SELECTED) -> List[Dict]:
    """
    Picks a capped subset for the LLM prompt, prioritizing category
    diversity (so Risk Analysis has real material for each risk type
    that has any coverage) over pure recency. The full, unfiltered
    list from fetch_company_news stays available separately for the
    transparency panel -- callers must not treat this selection as
    "everything that was retrieved."
    """
    if len(articles) <= max_count:
        return list(articles)

    selected = []
    selected_urls = set()

    for category in RISK_CATEGORIES:
        candidates = [a for a in articles if category in a["categories"]]
        for a in candidates[:PER_CATEGORY_MIN]:
            if a["url"] not in selected_urls:
                selected.append(a)
                selected_urls.add(a["url"])

    for a in articles:
        if len(selected) >= max_count:
            break
        if a["url"] not in selected_urls:
            selected.append(a)
            selected_urls.add(a["url"])

    return selected[:max_count]
