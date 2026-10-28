# AI-Powered Gift Card Recommendation System

- **Recommendation engine**: open-source LLM (default Llama 3.1 8B via Ollama; any OpenAI-compatible server works) with prompt engineering. The prompt restricts the model to a gift-card catalog and budget and treats user fields as data. Output is validated; hallucinated cards and over-budget amounts are dropped.
- **Distributed Redis cache**: normalized, hashed request keys; jittered TTL; a Redis `SET NX` lock with token-safe release so concurrent identical requests across instances make **one** LLM call (stampede protection).

## Setup
```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
docker compose up -d
docker compose exec ollama ollama pull llama3.1:8b
cp .env.example .env && set -a && . ./.env && set +a
.venv/bin/uvicorn app.main:get_app --factory --port 3000
curl -X POST localhost:3000/recommendations -H 'content-type: application/json' \
  -d '{"occasion":"birthday","interests":["music","gaming"],"budget":50}'
```
Response includes `source` (`llm`/`cache`) and an `X-Cache` header; `/stats` shows hit counts.

## Personalization
Requests may include `user_id` and `recipient_id`. Data lives in SQLite (`data/app.db`, override with `DB_PATH`).
- `PUT /users/{id}/profile` `{"interests": [...], "recipient_id": "mom"}` saves interests.
- `POST /feedback` `{"user_id","card_id","event":"purchase|click|dislike","recipient_id"}` records history.
- Candidates are ranked by **embedding similarity** (Ollama `nomic-embed-text`, `EMBED_MODEL`; falls back to keyword matching if the embedding server is down) + **global popularity** + the user's **category affinity**. Cards already gifted to / disliked by that recipient are removed.
- The profile summary goes into the prompt, and a profile fingerprint is part of the cache key, so users never share results across different histories (identical profiles still share).
Pull the embedding model once: `docker compose exec ollama ollama pull nomic-embed-text`.

## Catalog
By default cards come from `data/catalog.json` (~130 brands). Set `RELOADLY_CLIENT_ID`/`RELOADLY_CLIENT_SECRET` (free sandbox account at reloadly.com) to load products from the Reloadly gift-card sandbox instead (`RELOADLY_SANDBOX=false` for live). Either way, a keyword shortlist of ~40 candidates is sent to the LLM so large catalogs stay within the context window. Reloadly's response shape was taken from its public docs and is untested against the live sandbox.

## Test & benchmark
```sh
.venv/bin/pytest
.venv/bin/python scripts/bench.py   # set REDIS_URL for real Redis
```
The benchmark simulates LLM latency, so the improvement reflects the cache hit ratio, not a production measurement.
