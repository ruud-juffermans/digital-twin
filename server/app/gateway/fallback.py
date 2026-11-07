"""Fallback policy: answer with a second provider when the first one fails.

API failures are common enough that the post calls a fallback the gateway's
most direct reliability measure. The rule here is narrow on purpose: the
secondary only takes over when the primary failed before delivering a single
event (transport error, timeout, or an error status once the client's own
retries are exhausted). Once text has streamed to the visitor a switch would
duplicate it, so the error propagates instead. The meta event of a fallback
turn names the provider that answered and the one that failed, so cost and
spend are attributed to the right account.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .chat_model import OpenAICompatProvider
from .client import OpenAICompatError

logger = logging.getLogger(__name__)


class FallbackChatModel:
    def __init__(self, primary: OpenAICompatProvider, secondary: OpenAICompatProvider) -> None:
        self.primary = primary
        self.secondary = secondary

    @property
    def provider(self) -> str:
        return self.primary.provider

    @property
    def client(self) -> Any:
        """The primary's client (tests swap it the way they do on a plain model)."""
        return self.primary.client

    @client.setter
    def client(self, value: Any) -> None:
        self.primary.client = value

    async def stream(
        self, turns: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        delivered = False
        try:
            async for event in self.primary.stream(turns, **kwargs):
                delivered = True
                yield event
            return
        except (httpx.TransportError, OpenAICompatError) as exc:
            if delivered:
                raise
            logger.warning(
                "Provider %s failed before answering (%s); falling back to %s",
                self.primary.provider, exc, self.secondary.provider,
            )
        async for event in self.secondary.stream(turns, **kwargs):
            if event["type"] == "meta":
                event = {**event, "fallbackFrom": self.primary.provider}
            yield event

    async def close(self) -> None:
        await self.primary.close()
        await self.secondary.close()
