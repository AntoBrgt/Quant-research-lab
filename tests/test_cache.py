"""Cache key determinism and set/get roundtrip -- no network, no LLM."""

import cache


def test_same_input_same_key():
    k1 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v1")
    k2 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v1")
    assert k1 == k2


def test_whitespace_normalization_does_not_change_key():
    k1 = cache.build_cache_key("op", "hello   world", "model-a", "v1", "v1")
    k2 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v1")
    assert k1 == k2


def test_different_input_different_key():
    k1 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v1")
    k2 = cache.build_cache_key("op", "goodbye world", "model-a", "v1", "v1")
    assert k1 != k2


def test_different_model_different_key():
    k1 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v1")
    k2 = cache.build_cache_key("op", "hello world", "model-b", "v1", "v1")
    assert k1 != k2


def test_different_prompt_version_different_key():
    k1 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v1")
    k2 = cache.build_cache_key("op", "hello world", "model-a", "v2", "v1")
    assert k1 != k2


def test_different_schema_version_different_key():
    k1 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v1")
    k2 = cache.build_cache_key("op", "hello world", "model-a", "v1", "v2")
    assert k1 != k2


def test_ticker_alone_is_not_the_key_input():
    # A cache key built from the ticker string alone would collide across every
    # chunk for that ticker -- keys must be derived from actual content.
    k1 = cache.build_cache_key("op", "AAPL", "model-a", "v1", "v1")
    k2 = cache.build_cache_key("op", "some AAPL risk factors chunk text", "model-a", "v1", "v1")
    assert k1 != k2


def test_set_get_roundtrip():
    key = cache.build_cache_key("op", "content", "model-a", "v1", "v1")
    assert cache.get("ns", key) is None
    assert not cache.exists("ns", key)

    cache.set("ns", key, {"signals": [{"a": 1}]})

    assert cache.exists("ns", key)
    assert cache.get("ns", key) == {"signals": [{"a": 1}]}


def test_get_missing_key_returns_none():
    assert cache.get("ns", "does-not-exist") is None
