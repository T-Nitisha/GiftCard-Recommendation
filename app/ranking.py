import logging
import math

from .embeddings import cosine

log = logging.getLogger(__name__)


def card_text(c: dict) -> str:
    return f"{c['name']} ({c['category']}): {', '.join(c.get('tags', []))}"


def keyword_scores(cards: list[dict], interests: list[str], occasion: str | None) -> list[float]:
    words = {w for s in [*interests, occasion or ""] for w in s.lower().replace(",", " ").split() if len(w) > 2}
    scores = [sum(w in card_text(c).lower() for w in words) for c in cards]
    top = max(scores, default=0)
    return [s / top if top else 0.0 for s in scores]


def _norm_log(v: float, top: float) -> float:
    return math.log1p(max(v, 0)) / math.log1p(top) if top > 0 else 0.0


class Ranker:
    """Shortlists candidate cards: semantic similarity (embeddings, falling back to keywords)
    + global popularity + the user's category affinity; previously gifted/disliked cards are removed."""

    def __init__(self, embedder=None, w_sim=1.0, w_pop=0.15, w_cat=0.2):
        self.embedder, self.w_sim, self.w_pop, self.w_cat = embedder, w_sim, w_pop, w_cat

    async def rank(self, cards, req, profile: dict, popularity: dict, n: int = 40) -> list[dict]:
        interests = list(dict.fromkeys([*req.interests, *profile.get("interests", [])]))
        excluded = set(profile.get("gifted", [])) | set(profile.get("disliked", []))
        cards = [c for c in cards if c["id"] not in excluded]

        sims = await self._similarity(cards, req, interests)
        pop_top = max((popularity.get(c["id"], 0) for c in cards), default=0)
        aff = profile.get("category_affinity", {})
        aff_top = max(aff.values(), default=0)

        def score(i: int) -> float:
            c = cards[i]
            return (self.w_sim * sims[i]
                    + self.w_pop * _norm_log(popularity.get(c["id"], 0), pop_top)
                    + self.w_cat * (aff.get(c["category"], 0) / aff_top if aff_top else 0))

        order = sorted(range(len(cards)), key=score, reverse=True)
        return [cards[i] for i in order[:n]]

    async def _similarity(self, cards, req, interests) -> list[float]:
        if self.embedder and interests:
            query = f"{req.occasion or 'any occasion'} gift for {req.relationship or 'someone'} who likes {', '.join(interests)}"
            try:
                vecs = await self.embedder.embed([query, *map(card_text, cards)])
                return [max(cosine(vecs[0], v), 0.0) for v in vecs[1:]]
            except Exception as exc:  # embedding server down: degrade gracefully
                log.warning("embedding failed, using keyword matching: %s", exc)
        return keyword_scores(cards, interests, req.occasion)
