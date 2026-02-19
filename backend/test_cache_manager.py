from __future__ import annotations

import time
import unittest

from backend.cache_manager import CacheManager


class TestCacheManager(unittest.TestCase):
    def test_ttl_expiry(self) -> None:
        cache = CacheManager(max_items=10)
        cache.set("k", "v", ttl=1)
        self.assertEqual(cache.get("k"), "v")
        time.sleep(1.1)
        self.assertIsNone(cache.get("k"))

    def test_lru_eviction_when_capacity_exceeded(self) -> None:
        cache = CacheManager(max_items=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

    def test_get_refreshes_lru_order(self) -> None:
        cache = CacheManager(max_items=2)
        cache.set("a", 1)
        cache.set("b", 2)
        _ = cache.get("a")
        cache.set("c", 3)
        self.assertEqual(cache.get("a"), 1)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)


if __name__ == "__main__":
    unittest.main()
