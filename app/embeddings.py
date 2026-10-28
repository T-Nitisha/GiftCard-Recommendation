import math

from openai import AsyncOpenAI


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Embedder:
    """Embeddings from an OpenAI-compatible server (Ollama: nomic-embed-text, bge, etc.).
    Card vectors are cached in-process; batches keep large catalogs to a few requests."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", batch_size: int = 64):
        self.model, self.batch_size = model, batch_size
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed", max_retries=1, timeout=30)
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i:i + self.batch_size]
            res = await self.client.embeddings.create(model=self.model, input=chunk)
            for text, item in zip(chunk, res.data):
                self._cache[text] = item.embedding
        return [self._cache[t] for t in texts]
