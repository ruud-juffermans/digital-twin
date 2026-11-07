"""Product-feature tables: feedback, unanswered questions, and the contact
archive — on SQLite, plus the HTTP surfaces through the app."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.actions.approval import archive_contact_messages
from app.config import get_settings
from app.main import create_app
from app.observability.feedback import record_feedback
from app.observability.telemetry import TurnRecorder
from app.storage.models import Base, ContactMessageRow, FeedbackRow, UnansweredQuestionRow
from tests.test_approvals import ADMIN, DRAFT_INPUT, start_paused_thread


@pytest.fixture(autouse=True)
def admin_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sessions():
    engine = create_async_engine("sqlite+aiosqlite://")

    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(setup())
    yield async_sessionmaker(engine, expire_on_commit=False)
    asyncio.run(engine.dispose())


def rows(sessions, model):
    async def query():
        async with sessions() as session:
            return (await session.execute(select(model))).scalars().all()

    return asyncio.run(query())


def record(recorder, **overrides):
    kwargs = {
        "thread_id": "t-1",
        "outcome": "answered",
        "meta": None,
        "tools_used": [],
        "guard_flags": [],
        "visitor_message": "Where does Ruud work?",
        "answer": "Ruud runs an independent consulting practice.",
    }
    kwargs.update(overrides)
    asyncio.run(recorder.record(**kwargs))


def test_recorder_logs_unanswered_questions(sessions):
    recorder = TurnRecorder(sessions, provider="gemini", model="m")

    # A normal grounded answer: no row.
    record(recorder)
    # A guard refusal and an honest model-side "not in my knowledge base".
    record(
        recorder,
        outcome="refused",
        guard_flags=["input:offtopic"],
        visitor_message="Write me a scraper",
        answer="I can only help with questions about Ruud.",
    )
    record(
        recorder,
        visitor_message="What is Ruud's favorite color?",
        answer="The knowledge base doesn't mention Ruud's favorite color.",
        model_refusal=True,
    )

    unanswered = rows(sessions, UnansweredQuestionRow)
    assert [(row.reason, row.question) for row in unanswered] == [
        ("guard", "Write me a scraper"),
        ("model", "What is Ruud's favorite color?"),
    ]


def test_record_feedback_round_trip(sessions):
    asyncio.run(
        record_feedback(
            sessions,
            thread_id="t-1",
            verdict="down",
            question="Where did Ruud study?",
            answer="He studied Computer Science.",
            comment="Too vague.",
        )
    )

    (row,) = rows(sessions, FeedbackRow)
    assert (row.verdict, row.comment) == ("down", "Too vague.")


def test_archive_contact_messages_keeps_only_drafts(sessions):
    archived = asyncio.run(
        archive_contact_messages(
            sessions,
            thread_id="t-1",
            tool_calls=[
                {"name": "get_availability", "input": {}},
                {"name": "draft_contact_message", "input": DRAFT_INPUT},
            ],
            note="Sounds interesting.",
        )
    )

    assert archived == 1
    (row,) = rows(sessions, ContactMessageRow)
    assert row.subject == DRAFT_INPUT["subject"]
    assert row.sender_contact == DRAFT_INPUT["sender_contact"]
    assert row.note == "Sounds interesting."


def test_feedback_endpoint_requires_database():
    with TestClient(create_app()) as client:
        response = client.post("/api/feedback", json={"thread_id": "t-1", "verdict": "up"})
    assert response.status_code == 503


def test_feedback_endpoint_stores_row(sessions):
    with TestClient(create_app()) as client:
        client.app.state.sessions = sessions
        response = client.post(
            "/api/feedback",
            json={"thread_id": "t-1", "verdict": "down", "question": "Q", "answer": "A"},
        )
        invalid = client.post("/api/feedback", json={"thread_id": "t-1", "verdict": "meh"})

    assert response.status_code == 204
    assert invalid.status_code == 400
    (row,) = rows(sessions, FeedbackRow)
    assert (row.thread_id, row.verdict, row.question) == ("t-1", "down", "Q")


def test_approving_archives_contact_message(sessions):
    with TestClient(create_app()) as client:
        client.app.state.sessions = sessions
        start_paused_thread(client, "t-archive")
        response = client.post(
            "/api/admin/approvals/t-archive",
            json={"approved": True, "note": "Go ahead."},
            headers=ADMIN,
        )

    assert response.status_code == 200
    (row,) = rows(sessions, ContactMessageRow)
    assert (row.thread_id, row.subject, row.note) == (
        "t-archive", DRAFT_INPUT["subject"], "Go ahead.",
    )


def test_rejecting_archives_nothing(sessions):
    with TestClient(create_app()) as client:
        client.app.state.sessions = sessions
        start_paused_thread(client, "t-no-archive")
        response = client.post(
            "/api/admin/approvals/t-no-archive", json={"approved": False}, headers=ADMIN
        )

    assert response.status_code == 200
    assert rows(sessions, ContactMessageRow) == []
