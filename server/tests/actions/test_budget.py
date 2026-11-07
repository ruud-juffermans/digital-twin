import time

from fastapi.testclient import TestClient

from app.actions.budget import BudgetTracker
from app.guardrails.policies import BUDGET_REFUSAL
from app.main import create_app
from tests.conftest import fake_llm, parse_events, text_stream


def user_message(content: str = "Who is Ruud?") -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def test_budget_guard_fails_closed_and_tracks_spend():
    llm, recorder = fake_llm([text_stream("An answer.")])
    with TestClient(create_app()) as client:
        client.app.state.llm.client = llm

        # A normal turn spends budget (fixture: 100 in + 50 out).
        client.post("/api/chat", json=user_message())
        assert client.app.state.budget.spent_usd > 0
        assert recorder.calls == 1

        # Exceed the cap: the next turn must refuse before any model call.
        client.app.state.budget.add(999)
        response = client.post("/api/chat", json=user_message())

    events = parse_events(response.text)
    assert events[0] == {"type": "text", "text": BUDGET_REFUSAL}
    assert recorder.calls == 1  # no second call


def test_budget_tracker_disabled_when_zero():
    tracker = BudgetTracker(0)
    tracker.add(1000)
    assert tracker.exceeded is False


def test_budget_resets_at_midnight(monkeypatch):
    tracker = BudgetTracker(1.0)
    monkeypatch.setattr(time, "strftime", lambda _fmt: "2026-09-04")
    tracker.add(5.0)
    assert tracker.exceeded is True
    monkeypatch.setattr(time, "strftime", lambda _fmt: "2026-09-05")
    assert tracker.exceeded is False
    assert tracker.spent_usd == 0
