"""
crypto_resolver.py

Crypto ticker/name resolution, mirroring app/core/company_resolver.py's
US/NSE index pattern (fetch-once, cache-to-disk, TTL-refresh) but
scoped to CoinGecko's top 250 coins by market cap via /coins/markets --
NOT the full /coins/list (10,000+ entries, dominated by low-liquidity/
spam tokens). yfinance stays the sole price/quote source everywhere
else in this app; CoinGecko is used here ONLY to resolve a name/symbol
to yfinance's "{SYMBOL}-USD" ticker convention.

Deliberately NOT fuzzy/substring-matching all 250 coin NAMES against
free text, unlike company_resolver.py's title matching: several top-250
coins have short, common-English-word names (e.g. "Near", "Sei") that
would collide constantly with ordinary sentences ("questions near
retirement", "the sei[ze] of..."). Symbol matching (BTC, ETH, SOL --
short, distinctive, uppercase codes) is safe against the live top-250
list; NAME matching is instead a small, hand-curated alias table of
unambiguous, well-known names only -- the same shape and same caution
company_resolver.py's own ALIASES dict already uses for Google/Meta,
not a broad automated scan.
"""

import json
import re
import time
from typing import List

import requests

from app.core.paths import DATA_DIR
from app.core.retry import retry_on_transient_error

CACHE_DIR = DATA_DIR / "filings_cache"
CRYPTO_INDEX_CACHE = CACHE_DIR / "crypto_index.json"
CRYPTO_INDEX_TTL_SECONDS = 7 * 24 * 3600

COINGECKO_MARKETS_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"
)

# Hand-picked, unambiguous coin names only -- see module docstring for
# why this isn't generated from the full top-250 list automatically.
NAME_ALIASES = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "dogecoin": "DOGE-USD",
    "cardano": "ADA-USD",
    "solana": "SOL-USD",
    "ripple": "XRP-USD",
    "litecoin": "LTC-USD",
    "polkadot": "DOT-USD",
    "chainlink": "LINK-USD",
    "avalanche": "AVAX-USD",
    "polygon": "MATIC-USD",
    "shiba inu": "SHIB-USD",
    "tron": "TRX-USD",
    "uniswap": "UNI-USD",
    "stellar": "XLM-USD",
    "monero": "XMR-USD",
    "cosmos": "ATOM-USD",
}

# Case-sensitive bare uppercase token, same discipline as
# company_resolver.py's own _TICKER_TOKEN -- many top-250 crypto
# symbols are deliberately chosen to double as ordinary English words
# ("ONE" / Harmony, "APE" / ApeCoin, "ICE"), so this must only match a
# token the user actually TYPED in caps, not any lowercase word
# uppercased for comparison (that would turn "the ice melted" into a
# crypto mention). "$eth"/"$BTC" (a real, unambiguous crypto convention
# regardless of case) is the one exception, matched separately below.
_SYMBOL_TOKEN = re.compile(r"\b[A-Z]{2,6}\b")
_DOLLAR_SYMBOL_TOKEN = re.compile(r"\$([A-Za-z]{2,6})\b")

_crypto_index = None


def _fetch_crypto_index() -> dict:
    def _do():
        resp = requests.get(COINGECKO_MARKETS_URL, timeout=15)
        resp.raise_for_status()
        return resp

    raw = retry_on_transient_error(_do).json()

    symbol_to_ticker = {}
    for entry in raw:
        symbol = (entry.get("symbol") or "").upper()
        if not symbol or symbol in symbol_to_ticker:
            continue
        symbol_to_ticker[symbol] = f"{symbol}-USD"

    return {"symbols": symbol_to_ticker}


def _load_crypto_index() -> dict:
    global _crypto_index
    if _crypto_index is not None:
        return _crypto_index

    CACHE_DIR.mkdir(exist_ok=True)

    if CRYPTO_INDEX_CACHE.exists():
        age = time.time() - CRYPTO_INDEX_CACHE.stat().st_mtime
        if age < CRYPTO_INDEX_TTL_SECONDS:
            with open(CRYPTO_INDEX_CACHE, "r", encoding="utf-8") as f:
                _crypto_index = json.load(f)
                return _crypto_index

    try:
        data = _fetch_crypto_index()
    except (requests.RequestException, ValueError, KeyError):
        # Same degrade-gracefully convention as _load_nse_index --
        # CoinGecko being unreachable means "no crypto resolved this
        # run", not a broken request for anything else.
        data = {"symbols": {}}

    with open(CRYPTO_INDEX_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    _crypto_index = data
    return _crypto_index


def is_crypto_ticker(ticker: str) -> bool:
    """True for any yfinance "{SYMBOL}-USD" style ticker -- checked by
    suffix, not against the live index, so a ticker resolved a while
    ago (e.g. already in a user's watchlist) still reads as crypto even
    if that coin has since dropped out of the top-250 refresh."""
    return ticker.upper().endswith("-USD") and "-" in ticker


def resolve_crypto_mentions(question: str) -> List[str]:
    """Ordered, de-duplicated "{SYMBOL}-USD" tickers for crypto
    mentions in `question` -- a curated name-alias check (see
    NAME_ALIASES) plus exact-symbol-token matching against the live
    top-250 index. No fuzzy matching (see module docstring)."""
    found = []
    lowered = question.lower()

    for name, ticker in NAME_ALIASES.items():
        if name in lowered and ticker not in found:
            found.append(ticker)

    index = _load_crypto_index()
    symbols = index["symbols"]

    for token in _SYMBOL_TOKEN.findall(question):
        ticker = symbols.get(token)
        if ticker and ticker not in found:
            found.append(ticker)

    for token in _DOLLAR_SYMBOL_TOKEN.findall(question):
        ticker = symbols.get(token.upper())
        if ticker and ticker not in found:
            found.append(ticker)

    return found
