"""Embeddings over the same OpenAI-compatible protocol as chat.

Gemini and OpenAI expose `POST {base_url}/embeddings` with an identical
request/response shape, so one client covers both; DeepSeek and Kimi have
no embeddings endpoint, which is why EMBEDDING_PROVIDER is independent of
LLM_PROVIDER (chat on DeepSeek, embeddings on Gemini is a fine pairing).
`model_id` tags every stored vector so a provider/model switch re-embeds on
the next sync instead of serving mixed vector spaces.
"""

import httpx

from ..config import Settings
from .providers import PROVIDERS

EMBED_BATCH_SIZE = 64
TIMEOUT_S = 30.0


class OpenAICompatEmbeddings:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=TIMEOUT_S,
            transport=transport,
            headers={"authorization": f"Bearer {api_key}"},
        )

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors += await self._embed_batch(texts[start : start + EMBED_BATCH_SIZE])
        return vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post(
            "/embeddings", json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        # OpenAI numbers every item with `index` and may answer out of order;
        # Gemini omits the field and answers in input order. Honor the index
        # when present, fall back to position otherwise.
        items = response.json()["data"]
        ordered = sorted(enumerate(items), key=lambda pair: pair[1].get("index", pair[0]))
        return [item["embedding"] for _, item in ordered]

    async def close(self) -> None:
        await self._client.aclose()


def create_embedder(settings: Settings) -> OpenAICompatEmbeddings:
    provider = settings.embedding_provider
    spec = PROVIDERS[provider]
    return OpenAICompatEmbeddings(
        provider=provider,
        model=getattr(settings, f"{provider}_embedding_model"),
        api_key=getattr(settings, spec["key_field"]),
        base_url=spec["base_url"],
    )
