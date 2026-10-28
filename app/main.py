import os
import time
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .cache import RecommendationCache
from .catalog import catalog_from_env
from .embeddings import Embedder
from .llm import OpenSourceLLM
from .ranking import Ranker
from .recommender import RecommendationRequest, Recommender
from .store import EVENT_WEIGHTS, ProfileStore


class ProfileUpdate(BaseModel):
    interests: list[str] = Field(max_length=20)
    recipient_id: str = Field("", max_length=100)


class Feedback(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    card_id: str = Field(min_length=1, max_length=100)
    event: str = Field(pattern="^(" + "|".join(EVENT_WEIGHTS) + ")$")
    recipient_id: str = Field("", max_length=100)



def create_app(recommender: Recommender | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if recommender is None:
            redis = aioredis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            llm = OpenSourceLLM(
                base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                model=os.getenv("LLM_MODEL", "llama3.1:8b"),
                api_key=os.getenv("LLM_API_KEY", ""),
            )
            cache = RecommendationCache(redis, ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")))
            embedder = None
            if os.getenv("EMBED_MODEL", "nomic-embed-text"):
                embedder = Embedder(os.getenv("EMBED_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")),
                                    os.getenv("EMBED_MODEL", "nomic-embed-text"), os.getenv("LLM_API_KEY", ""))
            app.state.recommender = Recommender(
                llm, cache, catalog_from_env(), ProfileStore(os.getenv("DB_PATH", "data/app.db")), Ranker(embedder))
        else:
            app.state.recommender = recommender
        yield

    app = FastAPI(title="Gift Card Recommender", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        return app.state.recommender.cache.stats

    @app.post("/recommendations")
    async def recommendations(body: RecommendationRequest, response: Response):
        start = time.perf_counter()
        try:
            result = await app.state.recommender.recommend(body)
        except Exception:
            raise HTTPException(502, "recommendation_failed")
        response.headers["X-Cache"] = "HIT" if result["source"] == "cache" else "MISS"
        return {**result, "latencyMs": round((time.perf_counter() - start) * 1000)}

    @app.put("/users/{user_id}/profile")
    async def put_profile(user_id: str, body: ProfileUpdate):
        store = app.state.recommender.store
        await store.set_interests(user_id, body.recipient_id, body.interests)
        return await store.get_profile(user_id, body.recipient_id)

    @app.post("/feedback", status_code=201)
    async def feedback(body: Feedback):
        rec = app.state.recommender
        card = next((c for c in await rec.catalog_source.load() if c["id"] == body.card_id), None)
        if card is None:
            raise HTTPException(404, "unknown card_id")
        await rec.store.record_event(body.user_id, body.recipient_id, card["id"], card["category"], body.event)
        return {"status": "recorded"}

    return app


def get_app() -> FastAPI:
    return create_app()
