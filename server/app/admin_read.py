"""Read API for the admin panel: stats, the prompt log, the safety tables,
evals, the knowledge-base state and the system prompt.

Same bearer token as the approvals endpoints (ADMIN_TOKEN); everything is
read-only except marking an unanswered question handled. The panel reaches
these through its own proxy, so the token never meets a browser.

Aggregation happens in Python over the rows of the requested window: the
twin sees tens of turns a day, not thousands, and this keeps every query
dialect-neutral (tests run on SQLite, production on Postgres).
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from .admin import require_admin
from .config import get_settings
from .db import (
    BudgetLedgerRow,
    ContactMessageRow,
    EvalCaseRow,
    EvalRunRow,
    FeedbackRow,
    GuardIncidentRow,
    KnowledgeChunkRow,
    TurnLogRow,
    UnansweredQuestionRow,
)
from .llm import SYSTEM_PROMPT, knowledge_version, prompt_version

router = APIRouter(prefix="/api/admin")

OUTCOMES = ("answered", "refused", "pending_approval", "error")
DAYS = Query(default=30, ge=1, le=365)
LIMIT = Query(default=200, ge=1, le=1000)
# Annotated, not a shared default: FastAPI binds a Path() instance to the first
# parameter name it meets, so one instance cannot serve two routes.
RowId = Annotated[int, Path(ge=1)]
ThreadId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


# ─── helpers ──────────────────────────────────────────────────────────────────


def _utc(value: Any) -> datetime:
    """Rows come back naive from SQLite and aware from Postgres; treat both as UTC."""
    if value is None:
        return datetime.now(UTC)
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _iso(value: Any) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _day(value: Any) -> str:
    return _utc(value).date().isoformat()


def _no_database() -> JSONResponse:
    return JSONResponse(
        status_code=503, content={"error": "No database configured (DATABASE_URL)."}
    )


def _guard(request: Request) -> JSONResponse | None:
    if (denied := require_admin(request)) is not None:
        return denied
    if request.app.state.sessions is None:
        return _no_database()
    return None


async def _turns_since(
    sessions: async_sessionmaker, since: datetime, until: datetime | None = None
) -> list[TurnLogRow]:
    query = select(TurnLogRow).where(TurnLogRow.created_at >= since)
    if until is not None:
        query = query.where(TurnLogRow.created_at < until)
    async with sessions() as session:
        return list((await session.execute(query.order_by(TurnLogRow.created_at))).scalars())


async def _feedback_since(
    sessions: async_sessionmaker, since: datetime, until: datetime | None = None
) -> list[FeedbackRow]:
    query = select(FeedbackRow).where(FeedbackRow.created_at >= since)
    if until is not None:
        query = query.where(FeedbackRow.created_at < until)
    async with sessions() as session:
        return list((await session.execute(query.order_by(FeedbackRow.created_at))).scalars())


async def _feedback_for_threads(
    sessions: async_sessionmaker, thread_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Per-thread 👍/👎 counts and comments, for turns and thread summaries."""
    if not thread_ids:
        return {}
    async with sessions() as session:
        rows = (
            await session.execute(
                select(FeedbackRow).where(FeedbackRow.thread_id.in_(thread_ids))
            )
        ).scalars()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = out.setdefault(row.thread_id, {"up": 0, "down": 0, "comments": []})
        entry["up" if row.verdict == "up" else "down"] += 1
        if row.comment:
            entry["comments"].append(row.comment)
    return out


def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def _window_stats(
    turns: list[TurnLogRow], feedback: list[FeedbackRow], since: datetime, until: datetime
) -> dict[str, Any]:
    outcomes = dict.fromkeys(OUTCOMES, 0)
    for turn in turns:
        outcomes[turn.outcome] = outcomes.get(turn.outcome, 0) + 1
    cost = sum(turn.cost_usd for turn in turns)
    input_tokens = sum(turn.input_tokens for turn in turns)
    cached = sum(turn.cached_tokens for turn in turns)
    latencies = [turn.latency_ms for turn in turns if turn.latency_ms > 0]
    return {
        "since": _iso(since),
        "until": _iso(until),
        "turns": len(turns),
        "threads": len({turn.thread_id for turn in turns}),
        "outcomes": outcomes,
        "cost_usd": round(cost, 5),
        "cost_per_turn_usd": round(cost / len(turns), 5) if turns else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": sum(turn.output_tokens for turn in turns),
        "cached_tokens": cached,
        "cache_read_ratio": round(cached / input_tokens, 4) if input_tokens else 0.0,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "tool_use_rate": (
            round(sum(1 for turn in turns if turn.tools_used) / len(turns), 4) if turns else 0.0
        ),
        "feedback": {
            "up": sum(1 for row in feedback if row.verdict == "up"),
            "down": sum(1 for row in feedback if row.verdict == "down"),
        },
    }


def _turn_json(turn: TurnLogRow, feedback: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": turn.id,
        "created_at": _iso(turn.created_at),
        "thread_id": turn.thread_id,
        "provider": turn.provider,
        "model": turn.model,
        "outcome": turn.outcome,
        "input_tokens": turn.input_tokens,
        "output_tokens": turn.output_tokens,
        "cached_tokens": turn.cached_tokens,
        "cost_usd": turn.cost_usd,
        "latency_ms": turn.latency_ms,
        "tools_used": turn.tools_used or [],
        "guard_flags": turn.guard_flags or [],
        "visitor_message": turn.visitor_message,
        "answer": turn.answer,
        "citations": turn.citations,
        "tool_calls": turn.tool_calls,
        "retrieved_chunks": turn.retrieved_chunks,
        "prompt_version": turn.prompt_version,
        "kb_version": turn.kb_version,
        "locale": turn.locale,
        "embed": turn.embed,
        "feedback": feedback,
    }


class TurnFilters(BaseModel):
    q: str | None = None
    outcome: Literal["answered", "refused", "pending_approval", "error"] | None = None
    flag: str | None = None
    tool: str | None = None
    has_citations: bool | None = None
    feedback: Literal["up", "down"] | None = None
    prompt_version: str | None = None

    def matches(self, turn: TurnLogRow, thread_feedback: dict[str, Any] | None) -> bool:
        if self.outcome and turn.outcome != self.outcome:
            return False
        if self.flag and not any(
            flag == self.flag or flag.startswith(f"{self.flag}:") for flag in turn.guard_flags or []
        ):
            return False
        if self.tool and self.tool not in (turn.tools_used or []):
            return False
        if self.has_citations is not None and bool(turn.citations) != self.has_citations:
            return False
        if self.prompt_version and turn.prompt_version != self.prompt_version:
            return False
        if self.feedback and not (thread_feedback or {}).get(self.feedback):
            return False
        if self.q:
            needle = self.q.lower()
            haystack = f"{turn.visitor_message or ''}\n{turn.answer or ''}".lower()
            if needle not in haystack:
                return False
        return True


def _filters(
    q: str | None = Query(default=None, max_length=200),
    outcome: str | None = Query(default=None),
    flag: str | None = Query(default=None, max_length=80),
    tool: str | None = Query(default=None, max_length=80),
    has_citations: bool | None = Query(default=None),
    feedback: str | None = Query(default=None),
    prompt_version: str | None = Query(default=None, max_length=16),
) -> TurnFilters:
    return TurnFilters(
        q=q or None,
        outcome=outcome if outcome in OUTCOMES else None,
        flag=flag or None,
        tool=tool or None,
        has_citations=has_citations,
        feedback=feedback if feedback in ("up", "down") else None,
        prompt_version=prompt_version or None,
    )


# ─── stats ────────────────────────────────────────────────────────────────────


@router.get("/stats/overview")
async def stats_overview(request: Request, days: int = DAYS) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    settings = get_settings()
    state = request.app.state
    now = datetime.now(UTC)
    since = now - timedelta(days=days)
    previous_since = since - timedelta(days=days)
    turns = await _turns_since(state.sessions, since)
    previous = await _turns_since(state.sessions, previous_since, since)
    feedback = await _feedback_since(state.sessions, since)
    previous_feedback = await _feedback_since(state.sessions, previous_since, since)
    pending = await state.approvals.list()
    return {
        "days": days,
        "provider": settings.llm_provider,
        "model": settings.active_model,
        "budget": {
            "daily_limit_usd": settings.daily_budget_usd,
            "spent_today_usd": state.budget.spent_usd,
        },
        "pending_approvals": len(pending),
        "window": _window_stats(turns, feedback, since, now),
        "previous": _window_stats(previous, previous_feedback, previous_since, since),
    }


@router.get("/stats/timeseries")
async def stats_timeseries(request: Request, days: int = DAYS) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    now = datetime.now(UTC)
    since = now - timedelta(days=days - 1)
    since = since.replace(hour=0, minute=0, second=0, microsecond=0)
    turns = await _turns_since(request.app.state.sessions, since)
    by_day: dict[str, dict[str, Any]] = {}
    threads: dict[str, set[str]] = defaultdict(set)
    for turn in turns:
        day = _day(turn.created_at)
        entry = by_day.setdefault(day, {"day": day, "turns": 0, "cost_usd": 0.0, "refusals": 0})
        entry["turns"] += 1
        entry["cost_usd"] += turn.cost_usd
        entry["refusals"] += turn.outcome == "refused"
        threads[day].add(turn.thread_id)
    series = []
    for offset in range(days):
        day = (since + timedelta(days=offset)).date().isoformat()
        entry = by_day.get(day, {"day": day, "turns": 0, "cost_usd": 0.0, "refusals": 0})
        series.append(
            {**entry, "cost_usd": round(entry["cost_usd"], 5), "threads": len(threads.get(day, ()))}
        )
    return {"days": days, "series": series}


@router.get("/budget")
async def budget(request: Request, days: int = DAYS) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    state = request.app.state
    first_day = (datetime.now(UTC) - timedelta(days=days - 1)).date().isoformat()
    async with state.sessions() as session:
        rows = (
            await session.execute(
                select(BudgetLedgerRow)
                .where(BudgetLedgerRow.day >= first_day)
                .order_by(BudgetLedgerRow.day)
            )
        ).scalars()
        ledger = [
            {"day": row.day, "spent_usd": row.spent_usd, "updated_at": _iso(row.updated_at)}
            for row in rows
        ]
    return {
        "days": days,
        "daily_limit_usd": get_settings().daily_budget_usd,
        "spent_today_usd": state.budget.spent_usd,
        "ledger": ledger,
    }


# ─── prompt log ───────────────────────────────────────────────────────────────


@router.get("/turns")
async def turns(
    request: Request,
    days: int = DAYS,
    limit: int = LIMIT,
    offset: int = Query(default=0, ge=0),
) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    filters = _filters(**_filter_params(request))
    sessions = request.app.state.sessions
    rows = await _turns_since(sessions, datetime.now(UTC) - timedelta(days=days))
    rows.reverse()  # newest first
    feedback = await _feedback_for_threads(sessions, {row.thread_id for row in rows})
    matched = [row for row in rows if filters.matches(row, feedback.get(row.thread_id))]
    page = matched[offset : offset + limit]
    return {
        "total": len(matched),
        "offset": offset,
        "limit": limit,
        "items": [_turn_json(row, feedback.get(row.thread_id)) for row in page],
    }


def _filter_params(request: Request) -> dict[str, Any]:
    params = request.query_params
    has = params.get("has_citations")
    return {
        "q": params.get("q"),
        "outcome": params.get("outcome"),
        "flag": params.get("flag"),
        "tool": params.get("tool"),
        "has_citations": None if has is None else has.lower() in ("1", "true", "yes"),
        "feedback": params.get("feedback"),
        "prompt_version": params.get("prompt_version"),
    }


def _thread_summary(
    thread_id: str, rows: list[TurnLogRow], feedback: dict[str, Any] | None
) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    flags: list[str] = []
    tools: list[str] = []
    for row in rows:
        flags += [flag for flag in row.guard_flags or [] if flag not in flags]
        tools += [tool for tool in row.tools_used or [] if tool not in tools]
    answered = [row.answer for row in rows if row.answer]
    return {
        "thread_id": thread_id,
        "started_at": _iso(first.created_at),
        "last_at": _iso(last.created_at),
        "first_question": first.visitor_message,
        "last_answer": answered[-1] if answered else None,
        "turns": len(rows),
        "cost_usd": round(sum(row.cost_usd for row in rows), 5),
        "outcomes": [row.outcome for row in rows],
        "guard_flags": flags,
        "tools_used": tools,
        "has_citations": any(row.citations for row in rows),
        "locale": next((row.locale for row in rows if row.locale), None),
        "embed": next((row.embed for row in rows if row.embed is not None), None),
        "feedback": feedback,
    }


@router.get("/threads")
async def threads(request: Request, days: int = DAYS, limit: int = LIMIT) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    filters = _filters(**_filter_params(request))
    sessions = request.app.state.sessions
    rows = await _turns_since(sessions, datetime.now(UTC) - timedelta(days=days))
    by_thread: dict[str, list[TurnLogRow]] = defaultdict(list)
    for row in rows:
        by_thread[row.thread_id].append(row)
    feedback = await _feedback_for_threads(sessions, set(by_thread))
    matched = [
        (turns[-1].id, _thread_summary(thread_id, turns, feedback.get(thread_id)))
        for thread_id, turns in by_thread.items()
        if any(filters.matches(turn, feedback.get(thread_id)) for turn in turns)
    ]
    # Newest activity first; the last turn's id breaks ties within one second.
    matched.sort(key=lambda item: (item[1]["last_at"], item[0]), reverse=True)
    return {"items": [summary for _, summary in matched[:limit]]}


@router.get("/threads/{thread_id}")
async def thread(request: Request, thread_id: ThreadId) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    state = request.app.state
    async with state.sessions() as session:
        rows = list(
            (
                await session.execute(
                    select(TurnLogRow)
                    .where(TurnLogRow.thread_id == thread_id)
                    .order_by(TurnLogRow.created_at)
                )
            ).scalars()
        )
        feedback_rows = list(
            (
                await session.execute(
                    select(FeedbackRow)
                    .where(FeedbackRow.thread_id == thread_id)
                    .order_by(FeedbackRow.created_at)
                )
            ).scalars()
        )
    if not rows:
        return JSONResponse(status_code=404, content={"error": "Thread not found."})
    feedback = await _feedback_for_threads(state.sessions, {thread_id})
    return {
        "thread_id": thread_id,
        "turns": [_turn_json(row, feedback.get(thread_id)) for row in rows],
        "feedback": [
            {
                "id": row.id,
                "created_at": _iso(row.created_at),
                "verdict": row.verdict,
                "question": row.question,
                "answer": row.answer,
                "comment": row.comment,
            }
            for row in feedback_rows
        ],
        "pending_approval": await state.approvals.get(thread_id),
        "total_cost_usd": round(sum(row.cost_usd for row in rows), 5),
    }


# ─── safety ───────────────────────────────────────────────────────────────────


@router.get("/unanswered")
async def unanswered(
    request: Request,
    days: int = DAYS,
    limit: int = LIMIT,
    status: Literal["open", "handled"] | None = None,
) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    query = (
        select(UnansweredQuestionRow)
        .where(UnansweredQuestionRow.created_at >= datetime.now(UTC) - timedelta(days=days))
        .order_by(UnansweredQuestionRow.created_at.desc())
        .limit(limit)
    )
    if status is not None:
        query = query.where(UnansweredQuestionRow.status == status)
    async with request.app.state.sessions() as session:
        rows = (await session.execute(query)).scalars()
        return {"items": [_unanswered_json(row) for row in rows]}


def _unanswered_json(row: UnansweredQuestionRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "created_at": _iso(row.created_at),
        "thread_id": row.thread_id,
        "question": row.question,
        "answer": row.answer,
        "reason": row.reason,
        "status": row.status,
    }


class UnansweredStatus(BaseModel):
    status: Literal["open", "handled"]


@router.patch("/unanswered/{item_id}")
async def set_unanswered_status(request: Request, body: UnansweredStatus, item_id: RowId) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    async with request.app.state.sessions() as session:
        row = await session.get(UnansweredQuestionRow, item_id)
        if row is None:
            return JSONResponse(status_code=404, content={"error": "Not found."})
        row.status = body.status
        await session.commit()
        return {"item": _unanswered_json(row)}


@router.get("/guard-incidents")
async def guard_incidents(request: Request, days: int = DAYS, limit: int = LIMIT) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    async with request.app.state.sessions() as session:
        rows = (
            await session.execute(
                select(GuardIncidentRow)
                .where(GuardIncidentRow.created_at >= datetime.now(UTC) - timedelta(days=days))
                .order_by(GuardIncidentRow.created_at.desc())
                .limit(limit)
            )
        ).scalars()
        return {
            "items": [
                {
                    "id": row.id,
                    "created_at": _iso(row.created_at),
                    "thread_id": row.thread_id,
                    "flags": row.flags,
                    "visitor_message": row.visitor_message,
                }
                for row in rows
            ]
        }


@router.get("/feedback")
async def feedback(
    request: Request,
    days: int = DAYS,
    limit: int = LIMIT,
    verdict: Literal["up", "down"] | None = None,
) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    query = (
        select(FeedbackRow)
        .where(FeedbackRow.created_at >= datetime.now(UTC) - timedelta(days=days))
        .order_by(FeedbackRow.created_at.desc())
        .limit(limit)
    )
    if verdict is not None:
        query = query.where(FeedbackRow.verdict == verdict)
    async with request.app.state.sessions() as session:
        rows = (await session.execute(query)).scalars()
        return {
            "items": [
                {
                    "id": row.id,
                    "created_at": _iso(row.created_at),
                    "thread_id": row.thread_id,
                    "verdict": row.verdict,
                    "question": row.question,
                    "answer": row.answer,
                    "comment": row.comment,
                }
                for row in rows
            ]
        }


@router.get("/contact-messages")
async def contact_messages(request: Request, days: int = DAYS, limit: int = LIMIT) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    async with request.app.state.sessions() as session:
        rows = (
            await session.execute(
                select(ContactMessageRow)
                .where(ContactMessageRow.created_at >= datetime.now(UTC) - timedelta(days=days))
                .order_by(ContactMessageRow.created_at.desc())
                .limit(limit)
            )
        ).scalars()
        return {
            "items": [
                {
                    "id": row.id,
                    "created_at": _iso(row.created_at),
                    "thread_id": row.thread_id,
                    "subject": row.subject,
                    "message": row.message,
                    "sender_contact": row.sender_contact,
                    "note": row.note,
                }
                for row in rows
            ]
        }


# ─── evals ────────────────────────────────────────────────────────────────────


def _run_json(row: EvalRunRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "created_at": _iso(row.created_at),
        "provider": row.provider,
        "model": row.model,
        "met": row.met,
        "cost_usd": row.cost_usd,
        "summary": row.summary,
        "failure_injection": row.failure_injection,
    }


@router.get("/evals")
async def evals(request: Request, limit: int = Query(default=50, ge=1, le=500)) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    async with request.app.state.sessions() as session:
        rows = (
            await session.execute(
                select(EvalRunRow).order_by(EvalRunRow.created_at.desc()).limit(limit)
            )
        ).scalars()
        return {"runs": [_run_json(row) for row in rows]}


@router.get("/evals/{run_id}")
async def eval_run(request: Request, run_id: RowId) -> Any:
    if (denied := _guard(request)) is not None:
        return denied
    async with request.app.state.sessions() as session:
        run = await session.get(EvalRunRow, run_id)
        if run is None:
            return JSONResponse(status_code=404, content={"error": "Eval run not found."})
        cases = (
            await session.execute(
                select(EvalCaseRow).where(EvalCaseRow.run_id == run_id).order_by(EvalCaseRow.id)
            )
        ).scalars()
        return {
            "run": _run_json(run),
            "cases": [
                {
                    "id": case.id,
                    "dataset": case.dataset,
                    "case_id": case.case_id,
                    "passed": case.passed,
                    "detail": case.detail,
                    "tools": case.tools or [],
                    "errors": case.errors or [],
                    "cost_usd": case.cost_usd,
                    "latency_ms": case.latency_ms,
                }
                for case in cases
            ],
        }


# ─── knowledge base and prompt ────────────────────────────────────────────────


@router.get("/knowledge")
async def knowledge(request: Request) -> Any:
    if (denied := require_admin(request)) is not None:
        return denied
    state = request.app.state
    settings = get_settings()
    docs = [
        {"slug": doc.slug, "title": doc.title, "chars": len(doc.text)} for doc in state.knowledge
    ]
    chunks: dict[str, Any] = {"total": 0, "by_doc": [], "embedding_model": None, "last_sync": None}
    if state.sessions is not None:
        async with state.sessions() as session:
            grouped = await session.execute(
                select(KnowledgeChunkRow.doc_slug, func.count())
                .group_by(KnowledgeChunkRow.doc_slug)
                .order_by(KnowledgeChunkRow.doc_slug)
            )
            by_doc = [{"doc_slug": slug, "count": count} for slug, count in grouped]
            model, last_sync = (
                await session.execute(
                    select(
                        func.max(KnowledgeChunkRow.embedding_model),
                        func.max(KnowledgeChunkRow.updated_at),
                    )
                )
            ).one()
        chunks = {
            "total": sum(entry["count"] for entry in by_doc),
            "by_doc": by_doc,
            "embedding_model": model,
            "last_sync": _iso(last_sync) if last_sync else None,
        }
    return {
        "documents": docs,
        "chunks": chunks,
        "semantic_search": state.retriever is not None,
        "embedding_provider": settings.embedding_provider or None,
    }


@router.get("/prompt")
async def prompt(request: Request, days: int = Query(default=365, ge=1, le=365)) -> Any:
    if (denied := require_admin(request)) is not None:
        return denied
    state = request.app.state
    settings = get_settings()
    current_version = prompt_version()
    versions: list[dict[str, Any]] = []
    if state.sessions is not None:
        rows = await _turns_since(state.sessions, datetime.now(UTC) - timedelta(days=days))
        feedback = await _feedback_for_threads(state.sessions, {row.thread_id for row in rows})
        grouped: dict[str | None, list[TurnLogRow]] = defaultdict(list)
        for row in rows:
            grouped[row.prompt_version].append(row)
        for version, group in grouped.items():
            answered = sum(1 for row in group if row.outcome == "answered")
            refused = sum(1 for row in group if row.outcome == "refused")
            thread_ids = {row.thread_id for row in group}
            up = sum(feedback.get(thread_id, {}).get("up", 0) for thread_id in thread_ids)
            down = sum(feedback.get(thread_id, {}).get("down", 0) for thread_id in thread_ids)
            versions.append(
                {
                    "prompt_version": version,
                    "current": version == current_version,
                    "first_seen": _iso(group[0].created_at),
                    "last_seen": _iso(group[-1].created_at),
                    "turns": len(group),
                    "threads": len(thread_ids),
                    "answered": answered,
                    "refused": refused,
                    "answer_rate": round(answered / len(group), 4),
                    "refusal_rate": round(refused / len(group), 4),
                    "cost_usd": round(sum(row.cost_usd for row in group), 5),
                    "provider": group[-1].provider,
                    "model": group[-1].model,
                    "feedback": {"up": up, "down": down},
                }
            )
        versions.sort(key=lambda version: version["last_seen"], reverse=True)
    return {
        "current": {
            "prompt_version": current_version,
            "kb_version": knowledge_version(state.knowledge),
            "provider": settings.llm_provider,
            "model": settings.active_model,
            "text": SYSTEM_PROMPT,
        },
        "versions": versions,
    }
