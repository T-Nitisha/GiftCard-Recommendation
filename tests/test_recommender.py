import asyncio
import json

import fakeredis
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.catalog import ReloadlyCatalog, normalize_reloadly, LocalCatalog
from app.cache import RecommendationCache, cache_key
from app.main import create_app
from app.recommender import RecommendationRequest, Recommender, parse_recommendations

REPLY = json.dumps({"recommendations": [
    {"cardId": "spotify", "suggestedAmount": 50, "reason": "Loves music"},
    {"cardId": "fake-card", "suggestedAmount": 20, "reason": "hallucinated"},
    {"cardId": "steam", "suggestedAmount": 10, "reason": "Gamer"},
]})


class FakeLLM:
    def __init__(self, delay=0):
        self.calls, self.delay = 0, delay

    async def complete(self, messages):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return REPLY


def setup(delay=0):
    llm = FakeLLM(delay)
    cache = RecommendationCache(fakeredis.FakeAsyncRedis(), wait_s=0.005)
    return Recommender(llm, cache), llm


def test_filters_hallucinations_and_caps_budget():
    out = parse_recommendations(REPLY, RecommendationRequest(budget=30, count=3), LocalCatalog()._cards)
    assert [r["cardId"] for r in out] == ["spotify", "steam"]
    assert out[0]["suggestedAmount"] == 30


def test_cache_key_ignores_case_and_order():
    assert cache_key({"interests": ["Music", "games"], "occasion": "Birthday"}) == cache_key(
        {"occasion": "birthday ", "interests": ["GAMES", "music"]}
    )


async def test_second_request_served_from_cache():
    rec, llm = setup()
    a = await rec.recommend({"interests": ["music"], "budget": 50})
    b = await rec.recommend({"interests": ["music"], "budget": 50})
    assert (a["source"], b["source"], llm.calls) == ("llm", "cache", 1)


async def test_concurrent_identical_requests_single_llm_call():
    rec, llm = setup(delay=0.05)
    results = await asyncio.gather(*[rec.recommend({"interests": ["gaming"]}) for _ in range(20)])
    assert llm.calls == 1
    assert sum(r["source"] == "llm" for r in results) == 1


async def test_invalid_input_rejected():
    rec, _ = setup()
    with pytest.raises(ValidationError):
        await rec.recommend({"budget": -5})


def test_api_endpoint_and_cache_header():
    rec, _ = setup()
    with TestClient(create_app(rec)) as c:
        body = {"interests": ["music"]}
        assert c.post("/recommendations", json=body).headers["X-Cache"] == "MISS"
        assert c.post("/recommendations", json=body).headers["X-Cache"] == "HIT"
        assert c.post("/recommendations", json={"budget": -1}).status_code == 422


def test_normalize_reloadly_dedupes_by_brand():
    products = [
        {"productId": 1, "productName": "Amazon US", "brand": {"brandName": "Amazon"}, "category": {"name": "Shopping"}},
        {"productId": 2, "productName": "Amazon CA", "brand": {"brandName": "Amazon"}},
        {"productId": 3, "productName": "Netflix"},
    ]
    cards = normalize_reloadly(products)
    assert [c["name"] for c in cards] == ["Amazon", "Netflix"]
    assert cards[0]["id"] == "1" and cards[0]["category"] == "shopping"


async def test_reloadly_catalog_paginates_and_uses_sandbox():
    import httpx

    def handler(request: httpx.Request):
        if request.url.host == "auth.reloadly.com":
            assert json.loads(request.content)["audience"] == "https://giftcards-sandbox.reloadly.com"
            return httpx.Response(200, json={"access_token": "t"})
        page = int(request.url.params["page"])
        return httpx.Response(200, json={"totalPages": 2, "content": [
            {"productId": page, "productName": f"P{page}", "brand": {"brandName": f"B{page}"}}]})

    cat = ReloadlyCatalog("id", "secret", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert [c["name"] for c in await cat.load()] == ["B1", "B2"]
