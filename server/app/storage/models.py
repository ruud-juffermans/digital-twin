"""Every table the app owns, as SQLAlchemy models.

Privacy invariant: no table stores IPs or fingerprints; rows are keyed by the
anonymous thread_id only, and rate limiting stays in-memory. The LangGraph
checkpointer manages its own tables outside alembic.
"""

from typing import Any

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
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on Postgres, plain JSON elsewhere (tests run the models on SQLite).
JSONVariant = JSON().with_variant(JSONB(), "postgresql")
# pgvector on Postgres (dimension-free: the embedding_model column scopes each
# vector space); JSON on SQLite so the sync logic stays testable without Postgres.
EmbeddingVariant = JSON().with_variant(Vector(), "postgresql")


class Base(DeclarativeBase):
    pass


class PendingApprovalRow(Base):
    """High-risk actions awaiting Ruud's decision — survives restarts."""

    __tablename__ = "pending_approvals"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONVariant)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())


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
    # Performance telemetry (migration 0009), never purged: no visitor text.
    # Time from the request to the first text the visitor saw.
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Finish reason of the final model call ("stop", "length", "tool_calls", …).
    finish_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Failed attempts across the turn's model calls (transport errors, 429, 5xx).
    retries: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Worst problem of the turn, or null: one of ERROR_TYPES.
    error_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Run trace: [{"node", "ms"}] per graph node and
    # [{"node": "tool:<name>", "ms", "ok", "error_type"}] per tool call.
    trace: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONVariant, nullable=True)
    # Length metrics, kept after the text itself is purged. context_chars is
    # the size of the tool results fed back to the model (the dynamic context).
    question_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    answer_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)


# Fixed vocabulary for turn_log.error_type, most severe first.
ERROR_TYPES = (
    "timeout",        # the provider did not answer in time
    "rate_limited",   # 429 after retries
    "provider_5xx",   # provider-side failure after retries
    "tool_timeout",   # a tool hit its timeout (the turn may still have answered)
    "tool_failed",    # a tool raised
    "truncated",      # the model stopped at max_tokens (finish_reason "length")
    "other",          # anything else that ended the turn in an error
)


class ModelCallRow(Base):
    """One model round of a turn — the per-call breakdown the summed turn
    numbers cannot give (a tool-using turn is two or more calls)."""

    __tablename__ = "model_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    turn_id: Mapped[int] = mapped_column(
        ForeignKey("turn_log.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)  # 0-based round within the turn
    model: Mapped[str] = mapped_column(String(100))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    output_chars: Mapped[int] = mapped_column(Integer, default=0)


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

