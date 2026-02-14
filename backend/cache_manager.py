class CacheManager:
    """
    Simple cache wrapper to allow swapping backend easily (dict -> Redis).
    """

    def __init__(self):
        self._store = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value, ttl: int = None):
        # TTL currently unused, future Redis adapter can use this.
        self._store[key] = value

    def clear(self):
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


cache = CacheManager()
