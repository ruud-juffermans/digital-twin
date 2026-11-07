import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.gateway.client import OpenAICompatClient, OpenAICompatError
from tests.conftest import fake_transport, sse_bytes, text_stream


def make_client(transport: httpx.AsyncBaseTransport) -> OpenAICompatClient:
    return OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1", transport=transport
    )


def test_stream_parses_sse_chunks():
    client = make_client(fake_transport(text_stream("hi")))

    async def collect():
        return [c async for c in client.stream({"model": "m", "messages": []})]

    chunks = asyncio.run(collect())
    assert len(chunks) == 3  # [DONE] is consumed, never yielded
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
    assert chunks[-1]["usage"]["completion_tokens"] == 50


def test_error_status_raises():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": "no key"})
    )
    client = make_client(transport)

    async def run():
        async for _ in client.stream({"model": "m", "messages": []}):
            pass

    with pytest.raises(OpenAICompatError):
        asyncio.run(run())


def flaky_transport(
    failures: list[int | Exception], chunks: list[dict[str, Any]]
) -> tuple[httpx.MockTransport, SimpleNamespace]:
    """Fail the first len(failures) requests (a status code or an exception
    each), then stream `chunks`. Returns the transport and a call counter."""
    calls = SimpleNamespace(count=0)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.count += 1
        if calls.count <= len(failures):
            failure = failures[calls.count - 1]
            if isinstance(failure, int):
                return httpx.Response(failure, json={"error": "busy"})
            raise failure
        return httpx.Response(
            200, content=sse_bytes(chunks), headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler), calls


def collect(client: OpenAICompatClient) -> list[dict[str, Any]]:
    async def run():
        return [c async for c in client.stream({"model": "m", "messages": []})]

    return asyncio.run(run())


def test_transport_errors_and_5xx_are_retried():
    transport, calls = flaky_transport(
        [httpx.ConnectError("boom"), 503], text_stream("recovered")
    )
    client = OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1", transport=transport,
        max_retries=2, retry_base_delay=0,
    )

    chunks = collect(client)
    assert chunks[0]["choices"][0]["delta"]["content"] == "recovered"
    assert calls.count == 3  # two failures absorbed, third attempt streamed


def test_client_errors_are_not_retried():
    transport, calls = flaky_transport([401], text_stream("never"))
    client = OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1", transport=transport, retry_base_delay=0,
    )

    with pytest.raises(OpenAICompatError) as info:
        collect(client)
    assert info.value.status == 401
    assert calls.count == 1


def test_retries_exhaust_then_raise():
    transport, calls = flaky_transport([httpx.ConnectError("down")] * 3, text_stream("never"))
    client = OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1", transport=transport,
        max_retries=2, retry_base_delay=0,
    )

    with pytest.raises(httpx.ConnectError):
        collect(client)
    assert calls.count == 3  # 1 + max_retries
