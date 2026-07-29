"""
cache.py

Thin Redis wrapper shared by every cache call site in the pipeline.
Process-wide singleton client, mirroring the pattern already used for
the shared Qwen generator (llm_provider.py) and FinBERT pipeline
(finbert.py) -- constructed once, reused everywhere, instead of every
caller opening its own connection.

Two different caching strategies are used by callers of this module,
deliberately not collapsed into one:

1. Content-addressed (narrative_builder.py, valuation_pipeline.py):
   the cache key is a hash of every input that actually determines the
   output (the full LLM prompt; the ticker + financial statements +
   market cap + beta). A hit can never serve a result computed from
   different inputs than the ones being asked for right now -- if any
   input changes, the key changes, and it's a natural miss. TTL here
   is purely a storage/memory bound, not the correctness mechanism.

2. TTL-only (market_data_tool.py's statement fetch): there's no
   "input" to hash besides the ticker and wall-clock time -- the whole
   point is caching across time. Safe here specifically because it's
   scoped to the income/balance/cash-flow statements, which only
   change quarterly; current_price is deliberately never cached this
   way (see market_data_tool.py) since serving a stale price on a
   live research tool would be a real correctness bug, not just a
   staleness inconvenience.

Never breaks a report: if Redis isn't running or REDIS_URL is
unreachable, every function here degrades to a silent no-op (cache_get
returns None, cache_set does nothing) -- same "degrade gracefully"
convention already used for news_client.py/institutional_ratings.py
when their own dependencies are unavailable.
"""

import hashlib
import os
import pickle
from typing import Any, Optional

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Connection attempts are retried at most this often -- avoids a
# per-request connection-timeout stall (socket_connect_timeout below)
# on every single cache_get/cache_set call for the entire lifetime of
# a process where Redis simply isn't running (the common case for a
# local dev run or the deployed HF Space, which has no Redis).
_RECHECK_INTERVAL_SECONDS = 30

_client = None
_last_check_time = 0.0
_available = False


def _client_or_none():
    global _client, _last_check_time, _available

    import time
    now = time.time()

    if _client is not None or now - _last_check_time < _RECHECK_INTERVAL_SECONDS:
        return _client if _available else None

    _last_check_time = now
    try:
        import redis
        candidate = redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        candidate.ping()
        _client = candidate
        _available = True
        return _client
    except Exception:
        _client = None
        _available = False
        return None


def cache_available() -> bool:
    return _client_or_none() is not None


def make_key(namespace: str, *parts: Any) -> str:
    """
    Deterministic, debuggable key: `finsight:{namespace}:{first_part}:{hash_of_everything}`.
    Keeping the first part (typically a ticker) readable in the key
    itself makes `redis-cli --scan --pattern "finsight:valuation:AAPL:*"`
    useful for manual inspection, instead of every key being an opaque hash.
    """
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]
    readable_prefix = str(parts[0]) if parts else "na"
    return f"finsight:{namespace}:{readable_prefix}:{digest}"


def cache_get(key: str) -> Optional[Any]:
    client = _client_or_none()
    if client is None:
        return None
    try:
        raw = client.get(key)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return pickle.loads(raw)
    except Exception:
        # Corrupt/incompatible entry (e.g. a schema change since it was
        # written) -- treat as a miss rather than crashing the report.
        return None


def cache_set(key: str, value: Any, ttl_seconds: int) -> None:
    client = _client_or_none()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, pickle.dumps(value))
    except Exception:
        pass
