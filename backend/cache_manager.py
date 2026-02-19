from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any


class CacheManager:
    """In-memory cache with optional TTL and bounded size.

    Design goals:
    - Preserve existing `get/set/clear/len` interface.
    - Avoid unbounded memory growth under long-running API traffic.
    - Keep operations thread-safe for mixed async/threaded usage.
    """

    def __init__(self, max_items: int | None = None):
        try:
            default_max = int(os.getenv("CACHE_MAX_ITEMS", "512"))
        except (TypeError, ValueError):
            default_max = 512
        self._max_items = max_items if max_items is not None else max(1, default_max)
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = threading.RLock()

    def _now(self) -> float:
        return time.time()

    def _prune_expired_unlocked(self) -> None:
        now = self._now()
        expired_keys = [
            key
            for key, (_value, expires_at) in self._store.items()
            if expires_at is not None and expires_at <= now
        ]
        for key in expired_keys:
            self._store.pop(key, None)

    def _enforce_max_items_unlocked(self) -> None:
        while len(self._store) > self._max_items:
            self._store.popitem(last=False)

    def get(self, key: str):
        with self._lock:
            self._prune_expired_unlocked()
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and expires_at <= self._now():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value, ttl: int = None):
        expires_at: float | None = None
        if ttl is not None:
            try:
                ttl_int = int(ttl)
            except (TypeError, ValueError):
                ttl_int = 0
            if ttl_int > 0:
                expires_at = self._now() + float(ttl_int)
        with self._lock:
            self._prune_expired_unlocked()
            self._store[key] = (value, expires_at)
            self._store.move_to_end(key)
            self._enforce_max_items_unlocked()

    def clear(self):
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            self._prune_expired_unlocked()
            return len(self._store)


cache = CacheManager()
