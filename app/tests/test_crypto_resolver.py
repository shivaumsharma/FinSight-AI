"""
Unit tests for app/data/crypto_resolver.py. _load_crypto_index is
monkeypatched to canned data in every test -- no live CoinGecko call,
unlike company_resolver.py's own (excluded-from-CI) index tests. Same
monkeypatchable-private-function seam as test_nse_filings_client.py's
`_fetch_json` convention.
"""

from app.data import crypto_resolver as cr


def _fake_index():
    return {"symbols": {"BTC": "BTC-USD", "ETH": "ETH-USD", "ONE": "ONE-USD", "APE": "APE-USD"}}


def test_resolves_a_curated_name_alias_case_insensitively():
    assert cr.resolve_crypto_mentions("what's going on with Bitcoin today") == ["BTC-USD"]
    assert cr.resolve_crypto_mentions("should i buy bitcoin") == ["BTC-USD"]


def test_resolves_multiple_distinct_mentions_in_order(monkeypatch):
    monkeypatch.setattr(cr, "_load_crypto_index", _fake_index)
    result = cr.resolve_crypto_mentions("Compare Bitcoin and ETH")
    assert result == ["BTC-USD", "ETH-USD"]


def test_bare_uppercase_symbol_resolves(monkeypatch):
    monkeypatch.setattr(cr, "_load_crypto_index", _fake_index)
    assert cr.resolve_crypto_mentions("How is ETH doing this week?") == ["ETH-USD"]


def test_lowercase_common_word_matching_a_symbol_does_not_false_positive(monkeypatch):
    """The whole reason symbol matching is case-sensitive: ONE (Harmony)
    and APE (ApeCoin) are real top-250 crypto symbols that are also
    ordinary English words -- typing them lowercase in an unrelated
    sentence must NOT resolve as a crypto mention."""
    monkeypatch.setattr(cr, "_load_crypto_index", _fake_index)
    assert cr.resolve_crypto_mentions("I want one of those apps someday") == []


def test_dollar_prefixed_symbol_resolves_regardless_of_case(monkeypatch):
    monkeypatch.setattr(cr, "_load_crypto_index", _fake_index)
    assert cr.resolve_crypto_mentions("hows $eth looking today") == ["ETH-USD"]
    assert cr.resolve_crypto_mentions("hows $BTC looking today") == ["BTC-USD"]


def test_unrecognized_bare_uppercase_token_is_ignored(monkeypatch):
    monkeypatch.setattr(cr, "_load_crypto_index", _fake_index)
    assert cr.resolve_crypto_mentions("What does DCF stand for") == []


def test_is_crypto_ticker():
    assert cr.is_crypto_ticker("BTC-USD") is True
    assert cr.is_crypto_ticker("btc-usd") is True
    assert cr.is_crypto_ticker("AAPL") is False
    assert cr.is_crypto_ticker("BAJFINANCE.NS") is False


def test_load_crypto_index_degrades_cleanly_on_fetch_failure(monkeypatch, tmp_path):
    import requests

    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no network")

    monkeypatch.setattr(cr, "_crypto_index", None)
    monkeypatch.setattr(cr, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cr, "CRYPTO_INDEX_CACHE", tmp_path / "crypto_index.json")
    monkeypatch.setattr(cr.requests, "get", _raise)

    index = cr._load_crypto_index()
    assert index == {"symbols": {}}
