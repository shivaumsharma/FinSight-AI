"""
Unit tests for app/core/cache.py.

No real Redis server is required (or running in CI/this dev machine)
-- these specifically verify the "Redis unreachable" no-op path never
raises, since that's the actual default state for a local dev run and
the deployed HF Space, and a caching layer that crashes a report when
its cache is unavailable would be worse than no cache at all.
"""

from app.core import cache


def test_make_key_is_deterministic_for_identical_inputs():
    key1 = cache.make_key("valuation", "AAPL", "some-hash-of-financials")
    key2 = cache.make_key("valuation", "AAPL", "some-hash-of-financials")
    assert key1 == key2


def test_make_key_differs_when_any_input_changes():
    key1 = cache.make_key("valuation", "AAPL", "financials-v1")
    key2 = cache.make_key("valuation", "AAPL", "financials-v2")
    assert key1 != key2


def test_make_key_includes_readable_namespace_and_ticker():
    key = cache.make_key("narrative", "AAPL", "prompt-hash")
    assert key.startswith("finsight:narrative:AAPL:")


def test_cache_get_returns_none_without_raising_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr(cache, "_client_or_none", lambda: None)
    assert cache.cache_get("finsight:test:AAPL:whatever") is None


def test_cache_set_is_a_silent_noop_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr(cache, "_client_or_none", lambda: None)
    # Must not raise -- a report's cache_set call should never be able
    # to break the report itself.
    cache.cache_set("finsight:test:AAPL:whatever", {"some": "value"}, ttl_seconds=60)


def test_cache_available_reflects_client_state(monkeypatch):
    monkeypatch.setattr(cache, "_client_or_none", lambda: None)
    assert cache.cache_available() is False

    monkeypatch.setattr(cache, "_client_or_none", lambda: object())
    assert cache.cache_available() is True


def test_round_trip_with_a_fake_in_memory_client(monkeypatch):
    """Exercises cache_get/cache_set's actual get/setex/pickle logic
    (not just the no-op path) against a minimal fake client, without
    needing a real Redis server."""

    store = {}

    class _FakeClient:
        def get(self, key):
            return store.get(key)

        def setex(self, key, ttl, value):
            store[key] = value

    monkeypatch.setattr(cache, "_client_or_none", lambda: _FakeClient())

    key = cache.make_key("test", "AAPL", "v1")
    assert cache.cache_get(key) is None

    cache.cache_set(key, {"wacc": 0.09}, ttl_seconds=60)
    assert cache.cache_get(key) == {"wacc": 0.09}
