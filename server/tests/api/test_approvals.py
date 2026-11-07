import pytest
from fastapi.testclient import TestClient

from app.api.chat import APPROVAL_NOTICE
from app.config import get_settings
from app.main import create_app
from tests.conftest import fake_llm, parse_events, text_stream, tool_call_stream

ADMIN = {"Authorization": "Bearer test-admin-token"}

DRAFT_INPUT = {
    "subject": "Freelance inquiry",
    "message": "I'd like to hire Ruud for a data platform project.",
    "sender_contact": "jane@example.com",
}


@pytest.fixture(autouse=True)
def admin_token(monkeypatch):
    # Settings are cached per process; clear around the override.
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def start_paused_thread(client: TestClient, thread_id: str):
    """Drive a chat to the approval interrupt; returns the request recorder."""
    llm, recorder = fake_llm([
        tool_call_stream("draft_contact_message", DRAFT_INPUT),
        text_stream("Your message has been sent to Ruud."),
    ])
    client.app.state.llm.client = llm
    response = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "Please pass my inquiry on to Ruud."}],
        "thread_id": thread_id,
    })
    assert response.status_code == 200
    events = parse_events(response.text)
    assert {"type": "approval", "status": "pending"} in events
    assert any(e.get("text") == APPROVAL_NOTICE for e in events)
    assert recorder.calls == 1  # paused before any execution or second model call
    return recorder


def test_high_risk_tool_pauses_and_lists_pending():
    with TestClient(create_app()) as client:
        start_paused_thread(client, "t-pending")
        response = client.get("/api/admin/approvals", headers=ADMIN)

    assert response.status_code == 200
    (pending,) = response.json()["pending"]
    assert pending["thread_id"] == "t-pending"
    assert pending["tool_calls"] == [{"name": "draft_contact_message", "input": DRAFT_INPUT}]
    assert pending["visitor_message"] == "Please pass my inquiry on to Ruud."


def test_approving_resumes_executes_and_records_draft():
    config = {"configurable": {"thread_id": "t-approve"}}
    with TestClient(create_app()) as client:
        recorder = start_paused_thread(client, "t-approve")
        response = client.post(
            "/api/admin/approvals/t-approve", json={"approved": True}, headers=ADMIN,
        )
        assert client.get("/api/admin/approvals", headers=ADMIN).json()["pending"] == []
        snapshot = client.portal.call(client.app.state.agent.aget_state, config)

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert response.json()["answer"] == "Your message has been sent to Ruud."
    assert recorder.calls == 2  # resumed: tool executed, model finished the answer
    # The tool result echoed to the model reports success, and the draft is kept.
    tool_result = recorder.payloads[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert '"recorded"' in tool_result["content"]
    assert snapshot.values["contact_drafts"] == [
        {"tool": "draft_contact_message", "input": DRAFT_INPUT},
    ]


def test_rejecting_declines_without_executing():
    with TestClient(create_app()) as client:
        recorder = start_paused_thread(client, "t-reject")
        response = client.post(
            "/api/admin/approvals/t-reject",
            json={"approved": False, "note": "Not taking projects right now."},
            headers=ADMIN,
        )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    # The model was told the action was declined — it never executed.
    tool_result = recorder.payloads[1]["messages"][-1]
    assert '"declined"' in tool_result["content"]
    assert "Not taking projects right now." in tool_result["content"]


def test_admin_endpoints_require_token():
    with TestClient(create_app()) as client:
        assert client.get("/api/admin/approvals").status_code == 401
        wrong = {"Authorization": "Bearer wrong"}
        assert client.get("/api/admin/approvals", headers=wrong).status_code == 401
        # Authorized, but nothing pending on that thread.
        response = client.post("/api/admin/approvals/t-x", json={"approved": True}, headers=ADMIN)
        assert response.status_code == 404


def test_admin_endpoints_disabled_without_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        assert client.get("/api/admin/approvals", headers=ADMIN).status_code == 403
