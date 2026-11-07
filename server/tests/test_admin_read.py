"""The admin panel's read API: stats, prompt log, safety tables, evals,
knowledge and prompt — on SQLite through the app, with rows written by the
real recorder."""

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db import (
    Base,
    EvalRunRow,
    FeedbackRow,
    PostgresApprovalStore,
    TurnLogRow,
    TurnRecorder,
    UnansweredQuestionRow,
    purge_expired_text,
    record_eval_run,
    record_feedback,
)
from app.main import create_app
from tests.conftest import fake_llm, text_stream, tool_call_stream

ADMIN = {"Authorization": "Bearer test-admin-token"}
META = {"inputTokens": 100, "outputTokens": 50, "cachedTokens": 75, "costUsd": 0.0123,
        "latencyMs": 800}


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


def record(recorder, **overrides):
    kwargs = {
        "thread_id": "t-1",
        "outcome": "answered",
        "meta": META,
        "tools_used": [],
        "guard_flags": [],
        "visitor_message": "Is Ruud available?",
        "answer": "Yes — see Curriculum Vitae.",
        "citations": ["Curriculum Vitae"],
    }
    kwargs.update(overrides)
    asyncio.run(recorder.record(**kwargs))


def seed(sessions):
    recorder = TurnRecorder(sessions, provider="gemini", model="m", prompt_version="p1",
                            kb_version="k1")
    record(recorder)
    record(recorder, thread_id="t-1", visitor_message="And his rate?",
           answer="Rates depend on the project.", citations=[], tools_used=["get_availability"],
           tool_calls=[{"name": "get_availability", "input": {}, "result_chars": 42}])
    record(recorder, thread_id="t-2", outcome="refused", meta=None,
           guard_flags=["input:offtopic"], visitor_message="Write me a scraper",
           answer="I can only help with questions about Ruud.", citations=[])
    asyncio.run(record_feedback(sessions, thread_id="t-1", verdict="down",
                                question="And his rate?", answer="Rates depend on the project.",
                                comment="Too vague."))


@contextmanager
def client_with(sessions):
    """The app without DATABASE_URL, then handed the SQLite sessions."""
    with TestClient(create_app()) as client:
        client.app.state.sessions = sessions
        client.app.state.approvals = PostgresApprovalStore(sessions)
        yield client


def test_overview_and_timeseries(sessions):
    seed(sessions)
    with client_with(sessions) as client:
        overview = client.get("/api/admin/stats/overview?days=7", headers=ADMIN).json()
        series = client.get("/api/admin/stats/timeseries?days=3", headers=ADMIN).json()
        budget = client.get("/api/admin/budget?days=7", headers=ADMIN).json()

    window = overview["window"]
    assert (window["turns"], window["threads"]) == (3, 2)
    assert window["outcomes"] == {"answered": 2, "refused": 1, "pending_approval": 0, "error": 0}
    assert window["cost_usd"] == pytest.approx(0.0246)
    assert window["cache_read_ratio"] == pytest.approx(0.75)
    assert window["latency_p50_ms"] == 800
    assert window["tool_use_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert window["feedback"] == {"up": 0, "down": 1}
    assert overview["previous"]["turns"] == 0
    assert overview["provider"] == get_settings().llm_provider
    assert overview["pending_approvals"] == 0

    assert len(series["series"]) == 3
    today = series["series"][-1]
    assert (today["turns"], today["threads"], today["refusals"]) == (3, 2, 1)
    assert series["series"][0]["turns"] == 0

    assert budget["ledger"][0]["spent_usd"] == pytest.approx(0.0246)
    assert budget["daily_limit_usd"] == get_settings().daily_budget_usd


def test_turns_filters_and_threads(sessions):
    seed(sessions)
    with client_with(sessions) as client:
        turns = client.get("/api/admin/turns?days=7", headers=ADMIN).json()
        refused = client.get("/api/admin/turns?days=7&outcome=refused", headers=ADMIN).json()
        cited = client.get("/api/admin/turns?days=7&has_citations=true", headers=ADMIN).json()
        searched = client.get("/api/admin/turns?days=7&q=scraper", headers=ADMIN).json()
        thumbs = client.get("/api/admin/turns?days=7&feedback=down", headers=ADMIN).json()
        paged = client.get("/api/admin/turns?days=7&limit=1&offset=1", headers=ADMIN).json()
        threads = client.get("/api/admin/threads?days=7", headers=ADMIN).json()
        flagged = client.get("/api/admin/threads?days=7&flag=input", headers=ADMIN).json()
        thread = client.get("/api/admin/threads/t-1", headers=ADMIN).json()
        missing = client.get("/api/admin/threads/nope", headers=ADMIN)

    assert turns["total"] == 3
    assert turns["items"][0]["visitor_message"] == "Write me a scraper"  # newest first
    assert turns["items"][-1]["citations"] == [{"title": "Curriculum Vitae"}]
    assert turns["items"][-1]["prompt_version"] == "p1"
    assert [t["outcome"] for t in refused["items"]] == ["refused"]
    assert cited["total"] == 1
    assert searched["total"] == 1
    assert thumbs["total"] == 2  # both turns of the thumbed-down thread
    assert (paged["total"], len(paged["items"]), paged["offset"]) == (3, 1, 1)

    assert [t["thread_id"] for t in threads["items"]] == ["t-2", "t-1"]
    t1 = threads["items"][1]
    assert (t1["turns"], t1["first_question"], t1["has_citations"]) == (
        2, "Is Ruud available?", True,
    )
    assert t1["tools_used"] == ["get_availability"]
    assert t1["feedback"] == {"up": 0, "down": 1, "comments": ["Too vague."]}
    assert [t["thread_id"] for t in flagged["items"]] == ["t-2"]

    assert len(thread["turns"]) == 2
    assert thread["turns"][1]["tool_calls"] == [
        {"name": "get_availability", "input": {}, "result_chars": 42},
    ]
    assert thread["feedback"][0]["comment"] == "Too vague."
    assert thread["pending_approval"] is None
    assert thread["total_cost_usd"] == pytest.approx(0.0246)
    assert missing.status_code == 404


def test_safety_tables_and_unanswered_status(sessions):
    seed(sessions)
    with client_with(sessions) as client:
        unanswered = client.get("/api/admin/unanswered?days=7", headers=ADMIN).json()
        (item,) = unanswered["items"]
        assert (item["reason"], item["status"]) == ("guard", "open")
        patched = client.patch(
            f"/api/admin/unanswered/{item['id']}", json={"status": "handled"}, headers=ADMIN,
        )
        assert patched.json()["item"]["status"] == "handled"
        assert client.get("/api/admin/unanswered?days=7&status=open", headers=ADMIN).json() == {
            "items": [],
        }
        assert client.patch("/api/admin/unanswered/999", json={"status": "open"},
                            headers=ADMIN).status_code == 404
        incidents = client.get("/api/admin/guard-incidents?days=7", headers=ADMIN).json()
        feedback = client.get("/api/admin/feedback?days=7&verdict=down", headers=ADMIN).json()
        messages = client.get("/api/admin/contact-messages?days=7", headers=ADMIN).json()

    assert incidents["items"][0]["flags"] == ["input:offtopic"]
    assert feedback["items"][0]["comment"] == "Too vague."
    assert messages == {"items": []}


def test_evals_knowledge_and_prompt(sessions):
    seed(sessions)
    run_id = asyncio.run(record_eval_run(sessions, {
        "provider": "gemini", "model": "m", "datasets": [{
            "dataset": "tool_selection", "passed": 1, "total": 1, "accuracy": 1.0,
            "threshold": 0.9, "met": True, "costUsd": 0.01,
            "results": [{"id": "c1", "passed": True, "detail": {}, "tools": ["x"], "errors": [],
                         "costUsd": 0.01, "latencyMs": 5}],
        }],
    }))
    with client_with(sessions) as client:
        runs = client.get("/api/admin/evals", headers=ADMIN).json()
        run = client.get(f"/api/admin/evals/{run_id}", headers=ADMIN).json()
        assert client.get("/api/admin/evals/999", headers=ADMIN).status_code == 404
        knowledge = client.get("/api/admin/knowledge", headers=ADMIN).json()
        prompt = client.get("/api/admin/prompt", headers=ADMIN).json()

    assert runs["runs"][0]["id"] == run_id
    assert run["cases"][0]["case_id"] == "c1"
    assert knowledge["documents"] and knowledge["semantic_search"] is False
    assert knowledge["chunks"]["total"] == 0
    current = prompt["current"]
    assert len(current["prompt_version"]) == 8 and len(current["kb_version"]) == 8
    assert "Digital Twin" in current["text"]
    (version,) = prompt["versions"]
    assert (version["prompt_version"], version["turns"], version["refused"]) == ("p1", 3, 1)
    assert version["current"] is False  # seeded rows carry a made-up version
    assert version["feedback"] == {"up": 0, "down": 1}


def test_read_api_requires_token_and_database(sessions):
    with TestClient(create_app()) as client:
        assert client.get("/api/admin/stats/overview").status_code == 401
        # Authorized but no DATABASE_URL: the stats need a database.
        assert client.get("/api/admin/stats/overview", headers=ADMIN).status_code == 503
        # Knowledge and prompt work without one.
        assert client.get("/api/admin/knowledge", headers=ADMIN).status_code == 200
        assert client.get("/api/admin/prompt", headers=ADMIN).json()["versions"] == []


def test_chat_records_text_citations_tool_calls_and_origin(sessions):
    llm, _ = fake_llm([
        tool_call_stream("get_availability", {}),
        text_stream("Available now, see Curriculum Vitae."),
    ])
    with TestClient(create_app()) as client:
        client.app.state.llm.client = llm
        client.app.state.recorder = TurnRecorder(
            sessions, provider="gemini", model="m", prompt_version="pv", kb_version="kv",
        )
        client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Is Ruud available?"}],
            "thread_id": "t-rec", "locale": "nl-NL", "embed": True,
        })

    async def fetch():
        async with sessions() as session:
            return (await session.get(TurnLogRow, 1))

    row = asyncio.run(fetch())
    assert row.visitor_message == "Is Ruud available?"
    assert row.answer == "Available now, see Curriculum Vitae."
    assert row.citations == [{"title": "Curriculum Vitae"}]
    assert row.tool_calls[0]["name"] == "get_availability"
    assert row.tool_calls[0]["result_chars"] > 0
    assert row.retrieved_chunks == []
    assert (row.prompt_version, row.kb_version) == ("pv", "kv")
    assert (row.locale, row.embed) == ("nl-NL", True)


def test_retention_clears_old_text_but_keeps_numbers(sessions):
    seed(sessions)
    old = datetime.now(UTC) - timedelta(days=100)

    async def age_first_rows():
        async with sessions() as session:
            await session.execute(update(TurnLogRow).where(TurnLogRow.id == 1)
                                  .values(created_at=old))
            await session.execute(update(FeedbackRow).values(created_at=old))
            await session.commit()

    asyncio.run(age_first_rows())
    assert asyncio.run(purge_expired_text(sessions, 90)) == 1
    assert asyncio.run(purge_expired_text(sessions, 90)) == 0  # idempotent

    async def fetch():
        async with sessions() as session:
            return await session.get(TurnLogRow, 1), await session.get(TurnLogRow, 2), \
                await session.get(FeedbackRow, 1)

    aged, fresh, feedback = asyncio.run(fetch())
    assert (aged.visitor_message, aged.answer, aged.citations) == (None, None, None)
    assert aged.cost_usd == pytest.approx(0.0123)  # numbers stay
    assert fresh.visitor_message == "And his rate?"
    assert (feedback.question, feedback.comment) == (None, None)


def test_store_text_false_never_writes_text(sessions):
    recorder = TurnRecorder(sessions, provider="gemini", model="m", store_text=False)
    record(recorder)

    async def fetch():
        async with sessions() as session:
            return await session.get(TurnLogRow, 1)

    row = asyncio.run(fetch())
    assert (row.visitor_message, row.answer, row.citations) == (None, None, None)
    assert row.input_tokens == 100


def test_unanswered_default_status_is_open(sessions):
    seed(sessions)

    async def fetch():
        async with sessions() as session:
            return await session.get(UnansweredQuestionRow, 1)

    assert asyncio.run(fetch()).status == "open"
    assert EvalRunRow.__tablename__ == "eval_runs"
