"""Data platform: Postgres persistence behind small interfaces.

Postgres is the database of record. With DATABASE_URL unset (tests, quick
bare-metal dev) the app runs on in-memory fallbacks and nothing survives a
restart.

Privacy invariant: no table stores IPs or fingerprints — rows are keyed by
the anonymous thread_id only, and rate limiting stays in-memory.

Schema changes go through alembic (server/migrations/); `run_migrations` is
called at startup so a deploy is always at head.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .agent.guards import looks_like_refusal

logger = logging.getLogger(__name__)

SERVER_DIR = Path(__file__).resolve().parent.parent

# JSONB on Postgres, plain JSON elsewhere (tests run the models on SQLite).
JSONVariant = JSON().with_variant(JSONB(), "postgresql")
# pgvector on Postgres (dimension-free: the embedding_model column scopes each
# vector space); JSON on SQLite so the sync logic stays testable without Postgres.
EmbeddingVariant = JSON().with_variant(Vector(), "postgresql")


def sqlalchemy_url(database_url: str) -> str:
    """Pin the psycopg3 driver onto a plain postgresql:// URL."""
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


class Base(DeclarativeBase):
    pass

class PendingApprovalRow(Base):
    """High-risk actions awaiting Ruud's decision — survives restarts."""

    __tablename__ = "pending_approvals"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONVariant)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

def create_engine_and_sessions(database_url: str) -> tuple[AsyncEngine, async_sessionmaker]:
    engine = create_async_engine(sqlalchemy_url(database_url), pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def run_migrations(database_url: str) -> None:
    """Bring the schema to head. Sync (alembic is sync) — call via to_thread."""
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(SERVER_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(SERVER_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_url(database_url))
    command.upgrade(cfg, "head")


class TurnLogRow(Base):
    """One row per chat turn: tokens, cost, latency, tools, guard outcome."""

    __tablename__ = "turn_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(100))
    # "answered" | "refused" | "pending_approval" | "error"
    outcome: Mapped[str] = mapped_column(String(20))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    tools_used: Mapped[list[str]] = mapped_column(JSONVariant, default=list)
    guard_flags: Mapped[list[str]] = mapped_column(JSONVariant, default=list)
    # Prompt logging (migration 0008). The text columns are nulled by the
    # retention job after TEXT_RETENTION_DAYS; the numbers above stay.
    visitor_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # [{"title": str}] — the knowledge documents the answer cited.
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONVariant, nullable=True)
    # [{"name", "input", "result_chars"}] — every tool call of the turn.
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONVariant, nullable=True)
    # [{"document", "heading", "score"}] — what search_knowledge returned.
    retrieved_chunks: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONVariant, nullable=True
    )
    # Short hashes of the system prompt and the knowledge base that produced
    # the answer, so a prompt change is visible as a step in the stats.
    prompt_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    kb_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Where the question came from: the visitor's locale and whether the chat
    # ran embedded (widget on the portfolio) or standalone.
    locale: Mapped[str | None] = mapped_column(String(16), nullable=True)
    embed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class BudgetLedgerRow(Base):
    """Daily spend, durable across restarts — backs the hard budget cap."""

    __tablename__ = "budget_ledger"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)  # YYYY-MM-DD (local)
    spent_usd: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class GuardIncidentRow(Base):
    """Refusals and injection attempts with the offending input."""

    __tablename__ = "guard_incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    flags: Mapped[list[str]] = mapped_column(JSONVariant)
    visitor_message: Mapped[str] = mapped_column(Text)


class EvalRunRow(Base):
    """One eval-suite invocation: provider, thresholds met, total cost."""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(100))
    met: Mapped[bool] = mapped_column(Boolean)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    # Per-dataset summaries (accuracy, threshold, met) without the case rows.
    summary: Mapped[list[dict[str, Any]]] = mapped_column(JSONVariant)
    # {"injected": n, "recovered": n} for --failure-injection runs, else null.
    failure_injection: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)


class EvalCaseRow(Base):
    """One case of one eval run — history instead of overwriting report.json."""

    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True)
    dataset: Mapped[str] = mapped_column(String(40))
    case_id: Mapped[str] = mapped_column(String(80))
    passed: Mapped[bool] = mapped_column(Boolean)
    detail: Mapped[Any] = mapped_column(JSONVariant, nullable=True)
    tools: Mapped[list[str]] = mapped_column(JSONVariant, default=list)
    errors: Mapped[list[str]] = mapped_column(JSONVariant, default=list)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)


class FeedbackRow(Base):
    """👍/👎 per answer; thumbs-down rows carry the question + answer so they
    can become eval cases."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[str] = mapped_column(String(4))  # "up" | "down"
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)


class UnansweredQuestionRow(Base):
    """Every refusal with the question asked — content-gap analytics."""

    __tablename__ = "unanswered_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(String(10))  # "guard" | "model"
    # Triage: "open" until Ruud marks it "handled" (added the fact to the
    # knowledge base, or decided it is fine to refuse).
    status: Mapped[str] = mapped_column(String(10), default="open", server_default="open")


class ContactMessageRow(Base):
    """Archive of approved draft_contact_message payloads."""

    __tablename__ = "contact_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    subject: Mapped[str] = mapped_column(Text)
    message: Mapped[str] = mapped_column(Text)
    sender_contact: Mapped[str] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class ApprovalStore(Protocol):
    async def add(self, thread_id: str, payload: dict[str, Any]) -> None: ...
    async def get(self, thread_id: str) -> dict[str, Any] | None: ...
    async def list(self) -> list[dict[str, Any]]: ...
    async def remove(self, thread_id: str) -> None: ...


class InMemoryApprovalStore:
    """Fallback without a database; the checkpoints stay the source of truth."""

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, Any]] = {}

    async def add(self, thread_id: str, payload: dict[str, Any]) -> None:
        self._pending[thread_id] = {"thread_id": thread_id, **payload}

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._pending.get(thread_id)

    async def list(self) -> list[dict[str, Any]]:
        return list(self._pending.values())

    async def remove(self, thread_id: str) -> None:
        self._pending.pop(thread_id, None)


class PostgresApprovalStore:
    def __init__(self, sessions: async_sessionmaker) -> None:
        self._sessions = sessions

    async def add(self, thread_id: str, payload: dict[str, Any]) -> None:
        async with self._sessions() as session:
            await session.merge(PendingApprovalRow(thread_id=thread_id, payload=payload))
            await session.commit()

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        async with self._sessions() as session:
            row = await session.get(PendingApprovalRow, thread_id)
        return None if row is None else {"thread_id": row.thread_id, **row.payload}

    async def list(self) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            query = select(PendingApprovalRow).order_by(PendingApprovalRow.created_at)
            rows = (await session.execute(query)).scalars().all()
        return [{"thread_id": row.thread_id, **row.payload} for row in rows]

    async def remove(self, thread_id: str) -> None:
        async with self._sessions() as session:
            await session.execute(
                delete(PendingApprovalRow).where(PendingApprovalRow.thread_id == thread_id)
            )
            await session.commit()


class KnowledgeChunkRow(Base):
    """One embedded chunk of a knowledge document.

    content_hash covers the embedded text, so editing a paragraph re-embeds
    only that chunk on the next sync; embedding_model scopes the vector space
    so switching providers re-embeds everything instead of mixing spaces.
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_slug: Mapped[str] = mapped_column(String(80), index=True)
    doc_title: Mapped[str] = mapped_column(String(200))
    heading: Mapped[str] = mapped_column(String(200), default="")
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    embedding: Mapped[Any] = mapped_column(EmbeddingVariant)
    embedding_model: Mapped[str] = mapped_column(String(100), index=True)
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class TurnRecorder:
    """Writes the per-turn telemetry: turn log, budget ledger, guard incidents.

    Best-effort by design — a telemetry failure must never break the visitor's
    chat stream, so every error is logged and swallowed.
    """

    def __init__(
        self,
        sessions: async_sessionmaker,
        provider: str,
        model: str,
        *,
        prompt_version: str | None = None,
        kb_version: str | None = None,
        store_text: bool = True,
    ) -> None:
        self._sessions = sessions
        self._provider = provider
        self._model = model
        self._prompt_version = prompt_version
        self._kb_version = kb_version
        # TEXT_RETENTION_DAYS=0: never store question/answer text at all.
        self._store_text = store_text

    async def record(
        self,
        *,
        thread_id: str,
        outcome: str,
        meta: dict[str, Any] | None,
        tools_used: list[str],
        guard_flags: list[str],
        visitor_message: str,
        answer: str = "",
        citations: list[str] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        locale: str | None = None,
        embed: bool | None = None,
    ) -> None:
        meta = meta or {}
        cost = float(meta.get("costUsd") or 0.0)
        text_columns: dict[str, Any] = {}
        if self._store_text:
            text_columns = {
                "visitor_message": visitor_message,
                "answer": answer,
                "citations": [{"title": title} for title in citations or []],
                "tool_calls": tool_calls or [],
                "retrieved_chunks": retrieved_chunks or [],
            }
        try:
            async with self._sessions() as session:
                session.add(
                    TurnLogRow(
                        thread_id=thread_id,
                        provider=self._provider,
                        model=self._model,
                        outcome=outcome,
                        input_tokens=int(meta.get("inputTokens") or 0),
                        output_tokens=int(meta.get("outputTokens") or 0),
                        cached_tokens=int(meta.get("cachedTokens") or 0),
                        cost_usd=cost,
                        latency_ms=int(meta.get("latencyMs") or 0),
                        tools_used=tools_used,
                        guard_flags=guard_flags,
                        prompt_version=self._prompt_version,
                        kb_version=self._kb_version,
                        locale=locale,
                        embed=embed,
                        **text_columns,
                    )
                )
                if guard_flags:
                    session.add(
                        GuardIncidentRow(
                            thread_id=thread_id, flags=guard_flags, visitor_message=visitor_message
                        )
                    )
                # Content-gap analytics: a guard refusal and an honest
                # model-side "not in my knowledge base" both count.
                if outcome == "refused":
                    reason = "guard"
                elif outcome == "answered" and looks_like_refusal(answer):
                    reason = "model"
                else:
                    reason = None
                if reason is not None:
                    session.add(
                        UnansweredQuestionRow(
                            thread_id=thread_id,
                            question=visitor_message,
                            answer=answer,
                            reason=reason,
                        )
                    )
                if cost:
                    await self._add_spend(session, cost)
                await session.commit()
        except Exception:
            logger.exception("Failed to record turn telemetry for thread %s", thread_id)

    async def _add_spend(self, session: Any, cost: float) -> None:
        day = _today()
        result = await session.execute(
            update(BudgetLedgerRow)
            .where(BudgetLedgerRow.day == day)
            .values(spent_usd=BudgetLedgerRow.spent_usd + cost)
        )
        if result.rowcount == 0:
            # First turn of the day; a concurrent insert may win the race.
            try:
                async with session.begin_nested():
                    session.add(BudgetLedgerRow(day=day, spent_usd=cost))
            except IntegrityError:
                await session.execute(
                    update(BudgetLedgerRow)
                    .where(BudgetLedgerRow.day == day)
                    .values(spent_usd=BudgetLedgerRow.spent_usd + cost)
                )


async def purge_expired_text(sessions: async_sessionmaker, retention_days: int) -> int:
    """Null the free-text columns of turns and feedback older than the
    retention window (PRD: visitor text is kept TEXT_RETENTION_DAYS, numeric
    telemetry forever). Returns the number of turn rows cleared."""
    cutoff = datetime.now(UTC) - timedelta(days=max(retention_days, 0))
    async with sessions() as session:
        turns = await session.execute(
            update(TurnLogRow)
            .where(TurnLogRow.created_at < cutoff, TurnLogRow.visitor_message.is_not(None))
            .values(
                visitor_message=None,
                answer=None,
                citations=None,
                tool_calls=None,
                retrieved_chunks=None,
            )
        )
        await session.execute(
            update(FeedbackRow)
            .where(
                FeedbackRow.created_at < cutoff,
                (FeedbackRow.question.is_not(None))
                | (FeedbackRow.answer.is_not(None))
                | (FeedbackRow.comment.is_not(None)),
            )
            .values(question=None, answer=None, comment=None)
        )
        await session.commit()
    return turns.rowcount or 0


async def load_today_spent(sessions: async_sessionmaker) -> float:
    """Seed the in-memory budget tracker after a restart (restart-proof cap)."""
    async with sessions() as session:
        spent = await session.scalar(
            select(BudgetLedgerRow.spent_usd).where(BudgetLedgerRow.day == _today())
        )
    return float(spent or 0.0)


async def record_feedback(
    sessions: async_sessionmaker,
    *,
    thread_id: str,
    verdict: str,
    question: str | None,
    answer: str | None,
    comment: str | None,
) -> None:
    async with sessions() as session:
        session.add(
            FeedbackRow(
                thread_id=thread_id,
                verdict=verdict,
                question=question,
                answer=answer,
                comment=comment,
            )
        )
        await session.commit()


async def archive_contact_messages(
    sessions: async_sessionmaker,
    *,
    thread_id: str,
    tool_calls: list[dict[str, Any]],
    note: str | None,
) -> int:
    """Archive the draft_contact_message payloads of an approved action."""
    drafts = [
        call.get("input") or {}
        for call in tool_calls
        if call.get("name") == "draft_contact_message"
    ]
    async with sessions() as session:
        for draft in drafts:
            session.add(
                ContactMessageRow(
                    thread_id=thread_id,
                    subject=str(draft.get("subject") or ""),
                    message=str(draft.get("message") or ""),
                    sender_contact=str(draft.get("sender_contact") or ""),
                    note=note,
                )
            )
        await session.commit()
    return len(drafts)


async def record_eval_run(sessions: async_sessionmaker, report: dict[str, Any]) -> int:
    """Persist one eval-suite report: an eval_runs row plus one eval_cases row
    per case. Returns the run id. Raises on failure — an eval run whose
    history can't be written should fail loudly, unlike chat telemetry."""
    datasets = report.get("datasets") or []
    run = EvalRunRow(
        provider=report["provider"],
        model=report["model"],
        met=all(summary["met"] for summary in datasets) if datasets else False,
        cost_usd=round(sum(summary.get("costUsd") or 0.0 for summary in datasets), 5),
        summary=[
            {key: value for key, value in summary.items() if key != "results"}
            for summary in datasets
        ],
        failure_injection=report.get("failure_injection"),
    )
    async with sessions() as session:
        session.add(run)
        await session.flush()  # assigns run.id for the case rows
        for summary in datasets:
            for row in summary.get("results") or []:
                session.add(
                    EvalCaseRow(
                        run_id=run.id,
                        dataset=summary["dataset"],
                        case_id=row["id"],
                        passed=row["passed"],
                        detail=row.get("detail"),
                        tools=row.get("tools") or [],
                        errors=row.get("errors") or [],
                        cost_usd=row.get("costUsd") or 0.0,
                        latency_ms=row.get("latencyMs") or 0,
                    )
                )
        await session.commit()
    return run.id
