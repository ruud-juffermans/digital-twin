"""The gateway's fallback policy: a second provider answers when the first
fails before any text has streamed, and never after."""

import asyncio

import httpx
import pytest

from app.config import Settings
from app.gateway.chat_model import OpenAICompatProvider
from app.gateway.client import OpenAICompatClient, OpenAICompatError
from app.gateway.fallback import FallbackChatModel
from app.pipeline.provider import grounded_provider
from tests.conftest import fake_llm, sse_bytes, text_stream

SETTINGS = Settings(llm_provider="gemini", gemini_api_key="k", deepseek_api_key="k")
TURNS = [{"role": "user", "content": "Where is Ruud based?"}]


def failing_client(status: int | None = 503) -> OpenAICompatClient:
    """Every request fails: an error status, or a dropped connection when None."""

    def handler(request: httpx.Request) -> httpx.Response:
        if status is None:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(status, text="upstream unavailable")

    return OpenAICompatClient(
        api_key="k", base_url="https://primary.test/v1",
        transport=httpx.MockTransport(handler), max_retries=1, retry_base_delay=0,
    )


def model(client: OpenAICompatClient, provider: str) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        SETTINGS, system_prompt="You are a test.", titles=["About me"],
        client=client, provider=provider,
    )


def collect(chat) -> list[dict]:
    async def run():
        return [event async for event in chat.stream(TURNS)]

    return asyncio.run(run())


@pytest.mark.parametrize("status", [503, 429, None])
def test_secondary_answers_when_primary_fails_before_any_output(status):
    secondary_client, recorder = fake_llm([text_stream("Amsterdam.")])
    chat = FallbackChatModel(
        model(failing_client(status), "gemini"), model(secondary_client, "deepseek")
    )

    events = collect(chat)

    assert [e["text"] for e in events if e["type"] == "text"] == ["Amsterdam."]
    meta = next(e for e in events if e["type"] == "meta")
    assert (meta["provider"], meta["fallbackFrom"]) == ("deepseek", "gemini")
    assert recorder.calls == 1


def test_primary_success_never_touches_the_secondary():
    primary_client, _ = fake_llm([text_stream("Amsterdam.")])
    secondary_client, recorder = fake_llm([text_stream("never")])
    chat = FallbackChatModel(model(primary_client, "gemini"), model(secondary_client, "deepseek"))

    events = collect(chat)

    meta = next(e for e in events if e["type"] == "meta")
    assert meta["provider"] == "gemini" and "fallbackFrom" not in meta
    assert recorder.calls == 0


def test_no_fallback_once_text_has_streamed():
    """A switch after delivery would duplicate text: the error propagates."""
    chunks = text_stream("Amster")[:1]  # first delta, then the stream breaks

    def handler(request: httpx.Request) -> httpx.Response:
        body = sse_bytes(chunks)[:-len(b"data: [DONE]\n\n")] + b"data: {not json\n\n"
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    primary = model(OpenAICompatClient(api_key="k", base_url="https://p.test/v1",
                                       transport=httpx.MockTransport(handler)), "gemini")
    secondary_client, recorder = fake_llm([text_stream("never")])
    chat = FallbackChatModel(primary, model(secondary_client, "deepseek"))

    with pytest.raises(Exception, match=r"json|JSON|Expecting"):
        collect(chat)
    assert recorder.calls == 0


def test_non_retryable_status_still_falls_back_before_output():
    """A 401 on the primary is our key, not the visitor's problem: the site
    stays up on the secondary and the warning in the log names the cause."""
    secondary_client, recorder = fake_llm([text_stream("Amsterdam.")])
    chat = FallbackChatModel(
        model(failing_client(401), "gemini"), model(secondary_client, "deepseek")
    )

    events = collect(chat)

    assert [e["text"] for e in events if e["type"] == "text"] == ["Amsterdam."]
    assert recorder.calls == 1


def test_error_propagates_when_both_fail():
    chat = FallbackChatModel(
        model(failing_client(503), "gemini"), model(failing_client(502), "deepseek")
    )
    with pytest.raises(OpenAICompatError) as info:
        collect(chat)
    assert info.value.status == 502


def test_grounded_provider_wires_the_fallback_only_when_usable():
    docs: list = []
    plain = grounded_provider(Settings(llm_provider="gemini", gemini_api_key="k"), docs)
    assert isinstance(plain, OpenAICompatProvider)

    same = grounded_provider(
        Settings(llm_provider="gemini", gemini_api_key="k", fallback_provider="gemini"), docs
    )
    assert isinstance(same, OpenAICompatProvider)

    no_key = grounded_provider(
        Settings(
            llm_provider="gemini", gemini_api_key="k", deepseek_api_key="",
            fallback_provider="deepseek",
        ),
        docs,
    )
    assert isinstance(no_key, OpenAICompatProvider)

    wired = grounded_provider(SETTINGS.model_copy(update={"fallback_provider": "deepseek"}), docs)
    assert isinstance(wired, FallbackChatModel)
    assert (wired.provider, wired.secondary.provider) == ("gemini", "deepseek")
    asyncio.run(wired.close())
