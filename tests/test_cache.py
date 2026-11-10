from app.cache import PredictionCache

FEATURES = {"age": 50, "bmi": 30.5, "smoker": "never"}


class FakeClock:
    """Lets tests move time forward instantly instead of sleeping."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cache_miss_returns_none():
    cache = PredictionCache(max_size=10, ttl_seconds=60)

    assert cache.get(PredictionCache.make_key(FEATURES, "1.0")) is None


def test_cache_hit_returns_stored_value():
    cache = PredictionCache(max_size=10, ttl_seconds=60)
    key = PredictionCache.make_key(FEATURES, "1.0")

    cache.set(key, {"risk_level": "high"})

    assert cache.get(key) == {"risk_level": "high"}


def test_returned_value_is_a_copy():
    cache = PredictionCache(max_size=10, ttl_seconds=60)
    cache.set("key", {"recommendations": ["a"]})

    cache.get("key")["recommendations"].append("changed")

    assert cache.get("key") == {"recommendations": ["a"]}


def test_entry_expires_after_ttl():
    clock = FakeClock()
    cache = PredictionCache(max_size=10, ttl_seconds=60, clock=clock)
    cache.set("key", {"risk_level": "low"})

    clock.now = 59
    assert cache.get("key") is not None
    clock.now = 61
    assert cache.get("key") is None
    assert len(cache) == 0


def test_model_version_change_gives_a_different_key():
    old_key = PredictionCache.make_key(FEATURES, "1.0")
    new_key = PredictionCache.make_key(FEATURES, "2.0")
    cache = PredictionCache(max_size=10, ttl_seconds=60)
    cache.set(old_key, {"risk_level": "low"})

    assert old_key != new_key
    assert cache.get(new_key) is None


def test_key_is_normalised():
    reordered = {"smoker": "never", "bmi": 30.5, "age": 50.0}  # different order, 50 vs 50.0

    assert PredictionCache.make_key(FEATURES, "1.0") == PredictionCache.make_key(reordered, "1.0")
    assert PredictionCache.make_key(FEATURES, "1.0") != PredictionCache.make_key({**FEATURES, "age": 51}, "1.0")


def test_least_recently_used_entry_is_evicted_at_max_size():
    cache = PredictionCache(max_size=2, ttl_seconds=60)
    cache.set("a", {"value": 1})
    cache.set("b", {"value": 2})
    cache.get("a")  # "a" is now the most recently used, so "b" should be evicted next

    cache.set("c", {"value": 3})

    assert len(cache) == 2
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
