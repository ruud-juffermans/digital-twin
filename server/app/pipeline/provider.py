"""Compose a grounded chat model: the context layer's system prompt on the
gateway's chat completions client. The one place both are imported."""

from ..config import Settings
from ..context.knowledge import KnowledgeDoc
from ..context.prompt import knowledge_system_prompt
from ..gateway.chat_model import OpenAICompatProvider
from ..gateway.client import OpenAICompatClient


def grounded_provider(
    settings: Settings, docs: list[KnowledgeDoc], client: OpenAICompatClient | None = None
) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        settings,
        system_prompt=knowledge_system_prompt(docs),
        titles=[doc.title for doc in docs],
        client=client,
    )
