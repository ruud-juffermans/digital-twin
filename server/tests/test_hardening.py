from fastapi.testclient import TestClient

from app.gateway.client import OpenAICompatClient
from app.main import create_app
from tests.conftest import fake_transport, text_stream


def wire_fake_llm(client: TestClient) -> None:
    client.app.state.llm.client = OpenAICompatClient(
        api_key="k", base_url="https://example.test/v1",
        transport=fake_transport(text_stream("ok")),
    )


def test_last_message_must_be_from_user():
    with TestClient(create_app()) as client:
        response = client.post("/api/chat", json={
            "messages": [{"role": "assistant", "content": "hi"}]
        })
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid messages payload."}


def test_invalid_payload_is_clean_json():
    with TestClient(create_app()) as client:
        response = client.post("/api/chat", json={"messages": "nope"})
    assert response.status_code == 400
    assert "Traceback" not in response.text


def test_rate_limit_returns_friendly_429():
    with TestClient(create_app()) as client:
        wire_fake_llm(client)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        for _ in range(20):
            assert client.post("/api/chat", json=payload).status_code == 200
        response = client.post("/api/chat", json=payload)
    assert response.status_code == 429
    assert response.json() == {"error": "Rate limit reached — try again in a few minutes."}
