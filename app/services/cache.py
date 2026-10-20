"""热点缓存：Redis 优先，未配置或不可达时退到进程内 LRU + TTL。"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import settings

PREFIX = "zhuan:qa:"


class Cache:
    def __init__(self) -> None:
        self.backend = "memory-lru"
        self._client = None
        self._lock = threading.RLock()
        self._lru: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._init()

    def _init(self) -> None:
        if not settings.redis_url:
            return
        try:
            import redis

            c = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1.0, socket_timeout=1.0, decode_responses=True)
            c.ping()
            self._client = c
            self.backend = "redis"
        except Exception:
            self._client = None
            self.backend = "memory-lru"

    @staticmethod
    def key(question: str, category: str = "") -> str:
        h = hashlib.md5(f"{category}|{question.strip()}".encode("utf-8")).hexdigest()
        return f"{PREFIX}{h}"

    def get(self, key: str) -> Any | None:
        if self._client is not None:
            try:
                raw = self._client.get(key)
                self.hits += 1 if raw else 0
                self.misses += 0 if raw else 1
                return json.loads(raw) if raw else None
            except Exception:
                self._client = None
                self.backend = "memory-lru"
        with self._lock:
            item = self._lru.get(key)
            if not item:
                self.misses += 1
                return None
            exp, raw = item
            if exp < time.time():
                self._lru.pop(key, None)
                self.misses += 1
                return None
            self._lru.move_to_end(key)
            self.hits += 1
            return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or settings.cache_ttl
        raw = json.dumps(value, ensure_ascii=False)
        if self._client is not None:
            try:
                self._client.setex(key, ttl, raw)
                return
            except Exception:
                self._client = None
                self.backend = "memory-lru"
        with self._lock:
            self._lru[key] = (time.time() + ttl, raw)
            self._lru.move_to_end(key)
            while len(self._lru) > settings.cache_max_items:
                self._lru.popitem(last=False)

    def clear(self) -> None:
        if self._client is not None:
            try:
                for k in self._client.scan_iter(f"{PREFIX}*"):
                    self._client.delete(k)
            except Exception:
                pass
        with self._lock:
            self._lru.clear()
        self.hits = self.misses = 0

    def status(self) -> dict:
        total = self.hits + self.misses
        return {
            "backend": self.backend,
            "items": len(self._lru) if self._client is None else None,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "ttl": settings.cache_ttl,
        }


_cache: Cache | None = None


def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache
