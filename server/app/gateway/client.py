"""Minimal raw Chat Completions client: auth, SSE parsing, retry/backoff.

Gemini, OpenAI, DeepSeek, and Kimi all speak this protocol. The stream
yields the raw chunk dicts (choices[].delta pieces, then a final chunk
carrying usage). Interpreting them is the provider layer's job.

Transient failures — connection errors, timeouts, 429/5xx — are retried
with exponential backoff, but only while nothing has been delivered yet:
once a chunk has gone out, a retry would duplicate text, so the error
propagates instead.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Transient statuses worth a retry; any other 4xx is our request or our key.
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class OpenAICompatError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUSES


class OpenAICompatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ):
        self._headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=120, transport=transport
        )
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    async def stream(
        self, payload: dict[str, Any], stats: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw chunks. `stats`, when given, is filled in place with
        `retries`: how many attempts failed before this one (observability
        for the provider layer; a stream is per request, so no shared state)."""
        for attempt in range(self._max_retries + 1):
            if stats is not None:
                stats["retries"] = attempt
            delivered = False
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json={**payload, "stream": True},
                    headers=self._headers,
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")
                        raise OpenAICompatError(
                            f"API {response.status_code}: {body[:500]}",
                            status=response.status_code,
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            return
                        delivered = True
                        yield json.loads(data)
                return
            except (httpx.TransportError, OpenAICompatError) as exc:
                retryable = isinstance(exc, httpx.TransportError) or exc.retryable
                if delivered or not retryable or attempt == self._max_retries:
                    raise
                delay = self._retry_base_delay * 2**attempt
                logger.warning(
                    "Retrying after %s (attempt %d of %d, in %.1fs)",
                    exc, attempt + 1, self._max_retries, delay,
                )
                await asyncio.sleep(delay)

    async def close(self) -> None:
        await self._client.aclose()
