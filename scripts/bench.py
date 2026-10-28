"""Latency under concurrent load: no cache vs Redis cache (simulated LLM latency).
Env: LLM_LATENCY_MS=800 CONCURRENCY=100 DISTINCT=10 WARM_RATIO=0.5 REDIS_URL (else fakeredis)."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.cache import RecommendationCache  # noqa: E402
from app.recommender import Recommender  # noqa: E402

LAT = float(os.getenv("LLM_LATENCY_MS", 800)) / 1000
N = int(os.getenv("CONCURRENCY", 100))
DISTINCT = int(os.getenv("DISTINCT", 10))
WARM = float(os.getenv("WARM_RATIO", 0.5))  # fraction of distinct queries already cached
REPLY = json.dumps({"recommendations": [{"cardId": "amazon", "suggestedAmount": 25, "reason": "versatile"}]})


class SlowLLM:
    async def complete(self, messages):
        await asyncio.sleep(LAT)
        return REPLY


class NoCache:
    stats = {}

    async def get_or_compute(self, key, compute):
        return await compute(), "llm"


async def run(label, cache):
    rec = Recommender(SlowLLM(), cache)

    async def one(i):
        t = time.perf_counter()
        await rec.recommend({"interests": [f"topic-{i % DISTINCT}"]})
        return (time.perf_counter() - t) * 1000

    times = sorted(await asyncio.gather(*[one(i) for i in range(N)]))
    avg = sum(times) / len(times)
    print(f"{label:<10} avg={avg:.0f}ms p95={times[int(len(times) * 0.95)]:.0f}ms")
    return avg


async def main():
    if os.getenv("REDIS_URL"):
        import redis.asyncio as aioredis
        redis = aioredis.from_url(os.environ["REDIS_URL"])
    else:
        import fakeredis
        redis = fakeredis.FakeAsyncRedis()
    await redis.flushall()
    base = await run("no cache", NoCache())
    cache = RecommendationCache(redis)
    warm = Recommender(SlowLLM(), cache)
    for i in range(round(DISTINCT * WARM)):
        await warm.recommend({"interests": [f"topic-{i}"]})
    cached = await run("redis", cache)
    print(f"improvement: {(base - cached) / base * 100:.0f}%")


asyncio.run(main())
