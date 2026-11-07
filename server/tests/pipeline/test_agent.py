from fastapi.testclient import TestClient

from app.gateway.client import OpenAICompatClient
from app.main import create_app
from tests.conftest import fake_transport, parse_events, text_stream


def wire_fake_llm(client: TestClient, text: str = "An answer.") -> None:
    client.app.state.llm.client = OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1",
        transport=fake_transport(text_stream(text)),
    )


def user_message(content: str = "Who is Ruud?", **extra: object) -> dict:
    return {"messages": [{"role": "user", "content": content}], **extra}


def test_trace_event_reports_nodes():
    with TestClient(create_app()) as client:
        wire_fake_llm(client)
        response = client.post("/api/chat", json=user_message())

    events = parse_events(response.text)
    assert [e["type"] for e in events] == ["text", "meta", "trace", "done"]
    trace = events[2]
    assert [n["node"] for n in trace["nodes"]] == ["guard_input", "generate", "verify"]
    assert all(n["ms"] is not None for n in trace["nodes"][:-1])
    assert trace["guardFlags"] == []


def test_thread_state_accumulates_across_turns():
    config = {"configurable": {"thread_id": "visitor-1"}}
    with TestClient(create_app()) as client:
        wire_fake_llm(client)
        for _ in range(2):
            response = client.post("/api/chat", json=user_message(thread_id="visitor-1"))
            assert response.status_code == 200
        # The app's event loop runs in the TestClient's portal thread.
        snapshot = client.portal.call(client.app.state.agent.aget_state, config)

    assert snapshot.values["message_count"] == 2
    assert snapshot.values["total_cost_usd"] > 0


def test_invalid_thread_id_is_rejected():
    with TestClient(create_app()) as client:
        response = client.post("/api/chat", json=user_message(thread_id="bad thread id!"))
    assert response.status_code == 400
