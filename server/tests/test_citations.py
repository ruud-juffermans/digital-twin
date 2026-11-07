import asyncio
from typing import Any

from app.config import Settings
from app.gateway.client import OpenAICompatClient
from app.knowledge import KnowledgeDoc
from app.pipeline.provider import grounded_provider
from tests.conftest import fake_transport

DOCS = [
    KnowledgeDoc(title="About Me", slug="about", text="Hi."),
    KnowledgeDoc(title="Projects", slug="projects", text="Things."),
]


def split_stream(parts: list[str]) -> list[dict[str, Any]]:
    return [
        *({"choices": [{"index": 0, "delta": {"content": part},
                        "finish_reason": None}]} for part in parts),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ]


def collect_events(parts: list[str]) -> list[dict[str, Any]]:
    provider = grounded_provider(
        Settings(llm_provider="gemini", gemini_api_key="k"), DOCS,
        client=OpenAICompatClient(
            api_key="k", base_url="https://example.test/v1",
            transport=fake_transport(split_stream(parts)),
        ),
    )

    async def run() -> list[dict[str, Any]]:
        return [e async for e in provider.stream([{"role": "user", "content": "q"}])]

    return asyncio.run(run())


def test_citation_emitted_once_even_across_split_chunks():
    events = collect_events(["See About ", "Me for details. About Me has more."])
    citations = [e for e in events if e["type"] == "citation"]
    assert citations == [{"type": "citation", "title": "About Me"}]


def test_unknown_titles_are_never_cited():
    events = collect_events(["The CV document says so."])
    assert [e for e in events if e["type"] == "citation"] == []
