"""Compose a grounded chat model: the context layer's system prompt on the
gateway's chat completions client, with the gateway's fallback policy when a
second provider is configured. The one place these are imported together."""

import logging

from ..config import Settings
from ..context.knowledge import KnowledgeDoc
from ..context.prompt import knowledge_system_prompt
from ..gateway.chat_model import OpenAICompatProvider
from ..gateway.client import OpenAICompatClient
from ..gateway.fallback import FallbackChatModel
from ..gateway.providers import PROVIDERS

logger = logging.getLogger(__name__)


def grounded_provider(
    settings: Settings, docs: list[KnowledgeDoc], client: OpenAICompatClient | None = None
) -> OpenAICompatProvider | FallbackChatModel:
    system_prompt = knowledge_system_prompt(docs)
    titles = [doc.title for doc in docs]
    primary = OpenAICompatProvider(
        settings, system_prompt=system_prompt, titles=titles, client=client
    )
    fallback = settings.fallback_provider
    if not fallback or fallback == primary.provider:
        return primary
    if not getattr(settings, PROVIDERS[fallback]["key_field"]):
        logger.warning(
            "FALLBACK_PROVIDER=%s has no API key configured; fallback disabled", fallback
        )
        return primary
    secondary = OpenAICompatProvider(
        settings, system_prompt=system_prompt, titles=titles, provider=fallback
    )
    return FallbackChatModel(primary, secondary)
