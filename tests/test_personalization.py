import json
import re

import fakeredis
from fastapi.testclient import TestClient

from app.cache import RecommendationCache
from app.catalog import LocalCatalog
from app.main import create_app
from app.ranking import Ranker
from app.recommender import RecommendationRequest, Recommender
from app.store import ProfileStore

CARDS = LocalCatalog()._cards


class BagOfWordsEmbedder:
    """Deterministic stand-in for an embedding model."""
    VOCAB = ["music", "guitar", "games", "coffee", "books", "fitness", "sports"]

    async def embed(self, texts):
        return [[float(t.lower().count(w)) for w in self.VOCAB] for t in texts]


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    async def complete(self, messages):
        self.prompts.append(json.loads(messages[1]["content"]))
        ids = [c["id"] for c in self.prompts[-1]["catalog"][:3]]
        return json.dumps({"recommendations": [{"cardId": i, "suggestedAmount": 10, "reason": "r"} for i in ids]})


def make(embedder=None):
    llm, store = RecordingLLM(), ProfileStore(":memory:")
    cache = RecommendationCache(fakeredis.FakeAsyncRedis(), wait_s=0.005)
    return Recommender(llm, cache, store=store, ranker=Ranker(embedder)), llm, store


async def test_embedding_ranking_surfaces_semantic_matches():
    req = RecommendationRequest(interests=["guitar", "music"])
    top = await Ranker(BagOfWordsEmbedder()).rank(CARDS, req, {}, {}, n=3)
    assert "guitar-center" in [c["id"] for c in top]


async def test_embedding_failure_falls_back_to_keywords():
    class Broken:
        async def embed(self, texts):
            raise RuntimeError("down")

    req = RecommendationRequest(interests=["guitar"])
    top = await Ranker(Broken()).rank(CARDS, req, {}, {}, n=3)
    assert top[0]["id"] == "guitar-center"


async def test_popularity_and_affinity_break_ties():
    req = RecommendationRequest()  # no interests: similarity is a tie everywhere
    by_pop = await Ranker().rank(CARDS, req, {}, {"nike": 40, "steam": 5}, n=2)
    assert [c["id"] for c in by_pop] == ["nike", "steam"]
    by_aff = await Ranker().rank(CARDS, req, {"category_affinity": {"pets": 3}}, {}, n=3)
    assert {c["category"] for c in by_aff} == {"pets"}


async def test_gifted_cards_never_recommended_again():
    rec, llm, store = make()
    await store.record_event("u1", "mom", "spotify", "entertainment", "purchase")
    out = await rec.recommend({"user_id": "u1", "recipient_id": "mom", "interests": ["music"]})
    assert "spotify" not in [r["cardId"] for r in out["recommendations"]]
    assert "spotify" not in [c["id"] for c in llm.prompts[0]["catalog"]]
    assert llm.prompts[0]["profile"]["previously_gifted"] == ["spotify"]


async def test_cache_is_profile_aware():
    rec, llm, store = make()
    body = {"interests": ["music"]}
    await rec.recommend({**body, "user_id": "a"})
    await rec.recommend({**body, "user_id": "b"})  # same (empty) profile -> shared entry
    assert len(llm.prompts) == 1
    await store.record_event("a", "", "spotify", "entertainment", "purchase")
    again = await rec.recommend({**body, "user_id": "a"})  # profile changed -> fresh result
    assert again["source"] == "llm" and len(llm.prompts) == 2


def test_profile_and_feedback_endpoints():
    rec, llm, store = make()
    with TestClient(create_app(rec)) as c:
        assert c.put("/users/u1/profile", json={"interests": ["coffee"]}).json()["interests"] == ["coffee"]
        assert c.post("/feedback", json={"user_id": "u1", "card_id": "starbucks", "event": "purchase"}).status_code == 201
        assert c.post("/feedback", json={"user_id": "u1", "card_id": "nope", "event": "purchase"}).status_code == 404
        assert c.post("/feedback", json={"user_id": "u1", "card_id": "starbucks", "event": "bogus"}).status_code == 422
        r = c.post("/recommendations", json={"user_id": "u1"})
        assert r.status_code == 200
        assert "starbucks" not in [x["cardId"] for x in r.json()["recommendations"]]
