from openai import AsyncOpenAI


class OpenSourceLLM:
    """Any OpenAI-compatible server hosting an open-source model (Ollama, vLLM, llama.cpp)."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed", max_retries=2, timeout=60)

    async def complete(self, messages: list[dict]) -> str:
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        return res.choices[0].message.content or ""
