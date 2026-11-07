import json

from fastapi.testclient import TestClient

from app.actions.tools.get_availability import AVAILABILITY
from app.gateway.client import OpenAICompatClient
from app.main import create_app
from tests.conftest import (
    fake_llm,
    fake_transport,
    parse_events,
    text_stream,
    tool_call_stream,
    visible,
)


def test_chat_streams_text_meta_done():
    with TestClient(create_app()) as client:
        client.app.state.llm.client = OpenAICompatClient(
            api_key="k", base_url="https://example.test/v1",
            transport=fake_transport(text_stream("Hello there.")),
        )
        response = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Who is Ruud?"}]
        })

    events = visible(parse_events(response.text))
    assert [e["type"] for e in events] == ["text", "meta", "done"]
    assert events[0]["text"] == "Hello there."
    assert events[1]["inputTokens"] == 100
    assert events[1]["cachedTokens"] == 75
    assert events[1]["costUsd"] > 0
    assert events[1]["latencyMs"] >= 0

def test_invalid_payload_is_rejected():
    with TestClient(create_app()) as client:
        assert client.post("/api/chat", json={"messages": []}).status_code == 400

def user_message(content: str = "Who is Ruud?") -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def test_tool_loop_executes_tool_and_aggregates_meta():
    llm, recorder = fake_llm([
        tool_call_stream("get_availability", {}),
        text_stream("Ruud is available now."),
    ])
    with TestClient(create_app()) as client:
        client.app.state.llm.client = llm
        response = client.post("/api/chat", json=user_message("Is Ruud available for hire?"))

    events = visible(parse_events(response.text))
    assert [e["type"] for e in events] == ["tool", "text", "meta", "done"]
    assert events[0] == {"type": "tool", "name": "get_availability"}
    assert events[1]["text"] == "Ruud is available now."
    # One meta event, aggregated across both model rounds (100+100 in, 30+50 out).
    assert events[2]["inputTokens"] == 200
    assert events[2]["outputTokens"] == 80

    # Round 1 declared the tools; round 2 echoed the assistant tool call and
    # answered it with a role:"tool" message.
    first, second = recorder.payloads
    assert [t["function"]["name"] for t in first["tools"]] == [
        "search_knowledge", "get_availability", "draft_contact_message",
    ]
    assistant_turn, tool_result = second["messages"][-2], second["messages"][-1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["tool_calls"][0]["function"]["name"] == "get_availability"
    assert tool_result == {
        "role": "tool", "tool_call_id": "call_1", "content": json.dumps(AVAILABILITY),
    }


def test_tool_loop_stops_at_round_budget():
    # A model that keeps asking for tools: the deterministic round budget
    # must end the loop and still produce a final answer.
    llm, recorder = fake_llm([
        tool_call_stream("get_availability", {}, call_id="call_1"),
        tool_call_stream("get_availability", {}, call_id="call_2"),
        tool_call_stream("get_availability", {}, call_id="call_3"),
        text_stream("Done."),
    ])
    with TestClient(create_app()) as client:
        client.app.state.llm.client = llm
        response = client.post("/api/chat", json=user_message())

    events = visible(parse_events(response.text))
    assert [e["type"] for e in events] == ["tool", "tool", "tool", "text", "meta", "done"]
    assert recorder.calls == 4
    # Past the budget the model is forced to answer in text.
    assert recorder.payloads[-1]["tool_choice"] == "none"
