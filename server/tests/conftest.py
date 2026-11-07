import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.api.rate_limit import limiter
from app.gateway.client import OpenAICompatClient


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    limiter.reset()

def sse_bytes(chunks: list[dict[str, Any]]) -> bytes:
    """Encode chunk dicts as an OpenAI-style SSE response body."""
    body = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
    return body + b"data: [DONE]\n\n"


def text_stream(text: str) -> list[dict[str, Any]]:
    """A minimal complete chunk sequence for one text answer."""
    return [
        {"choices": [{"index": 0, "delta": {"content": text},
                      "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [],
         "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                   "prompt_tokens_details": {"cached_tokens": 75}}},
    ]


def fake_transport(chunks: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse_bytes(chunks),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)

def parse_events(body: str) -> list[dict[str, Any]]:
    return [json.loads(line[len("data: "):])
            for line in body.splitlines() if line.startswith("data: ")]


def visible(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop diagnostic trace events; most tests assert the visitor-facing stream."""
    return [e for e in events if e["type"] != "trace"]

def tool_call_stream(name: str, arguments: dict[str, Any],
                     call_id: str = "call_1") -> list[dict[str, Any]]:
    """A chunk sequence for a turn that ends in one tool call, with the
    JSON arguments split across two deltas the way providers stream them."""
    args = json.dumps(arguments)
    return [
        {"choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": 0, "id": call_id,
             "function": {"name": name, "arguments": args[:1]}}]}}]},
        {"choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": args[1:]}}]}}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 30}},
    ]

def fake_llm(rounds: list[list[dict[str, Any]]]) -> tuple[OpenAICompatClient, SimpleNamespace]:
    """A client whose Nth request answers rounds[N] (the last round repeats),
    plus a recorder with `.calls` and `.payloads` for asserting on requests."""
    recorder = SimpleNamespace(calls=0, payloads=[])

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.calls += 1
        recorder.payloads.append(json.loads(request.content))
        body = rounds[min(recorder.calls - 1, len(rounds) - 1)]
        return httpx.Response(
            200, content=sse_bytes(body), headers={"content-type": "text/event-stream"},
        )

    client = OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1", transport=httpx.MockTransport(handler),
    )
    return client, recorder
