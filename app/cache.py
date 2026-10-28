import asyncio
import hashlib
import json
import random
import time
import uuid
from typing import Any, Awaitable, Callable

_RELEASE = 'if redis.call("get", KEYS[1]) == ARGV[1] then return redis.call("del", KEYS[1]) else return 0 end'


def _canonical(v: Any) -> Any:
    if isinstance(v, list):
        return sorted((_canonical(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(v, dict):
        return {k: _canonical(v[k]) for k in sorted(v)}
    if isinstance(v, str):
        return v.strip().lower()
    return v


def cache_key(request: dict, prefix: str = "rec:v1") -> str:
    norm = json.dumps(_canonical(request), sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(norm.encode()).hexdigest()}"


class RecommendationCache:
    """Read-through Redis cache. A distributed SET NX lock lets concurrent identical
    requests (across instances) share a single LLM call (stampede protection).
    TTL is jittered to avoid mass expiry."""

    def __init__(self, redis, ttl_seconds=3600, lock_seconds=30, wait_s=0.05, max_wait_s=10.0):
        self.redis, self.ttl, self.lock_seconds = redis, ttl_seconds, lock_seconds
        self.wait_s, self.max_wait_s = wait_s, max_wait_s
        self.stats = {"hits": 0, "misses": 0, "coalesced": 0}

    async def get(self, key: str):
        raw = await self.redis.get(key)
        return json.loads(raw) if raw else None

    async def set(self, key: str, value) -> None:
        ttl = self.ttl + int(self.ttl * 0.1 * random.random())
        await self.redis.set(key, json.dumps(value), ex=ttl)

    async def get_or_compute(self, key: str, compute: Callable[[], Awaitable[Any]]):
        hit = await self.get(key)
        if hit is not None:
            self.stats["hits"] += 1
            return hit, "cache"

        lock_key, token = f"{key}:lock", uuid.uuid4().hex
        deadline = time.monotonic() + self.max_wait_s
        while True:
            if await self.redis.set(lock_key, token, ex=self.lock_seconds, nx=True):
                try:
                    again = await self.get(key)
                    if again is not None:
                        self.stats["hits"] += 1
                        return again, "cache"
                    self.stats["misses"] += 1
                    value = await compute()
                    await self.set(key, value)
                    return value, "llm"
                finally:
                    await self.redis.eval(_RELEASE, 1, lock_key, token)
            await asyncio.sleep(self.wait_s)
            filled = await self.get(key)
            if filled is not None:
                self.stats["coalesced"] += 1
                return filled, "cache"
            if time.monotonic() > deadline:  # lock holder slow/dead: fall back
                self.stats["misses"] += 1
                value = await compute()
                await self.set(key, value)
                return value, "llm"
