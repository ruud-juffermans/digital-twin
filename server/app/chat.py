import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_settings
from .gateway.client import OpenAICompatError
from .observability.feedback import record_feedback
from .observability.tracing import TOOL_ERROR_FAILED, TOOL_ERROR_TIMEOUT
from .rate_limit import limiter
from .schemas import ChatRequest, FeedbackRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

APPROVAL_NOTICE = (
    "\n\nThis request needs Ruud's personal approval — it has been queued for his review."
)


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.post("/feedback")
@limiter.limit(get_settings().rate_limit)
async def feedback(request: Request, payload: FeedbackRequest) -> Response:
    """👍/👎 per answer; needs the database (the row is the whole point)."""
    sessions = request.app.state.sessions
    if sessions is None:
        return JSONResponse(
            status_code=503, content={"error": "Feedback is not available right now."}
        )
    await record_feedback(
        sessions,
        thread_id=payload.thread_id,
        verdict=payload.verdict,
        question=payload.question,
        answer=payload.answer,
        comment=payload.comment,
    )
    return Response(status_code=204)


@router.post("/chat")
@limiter.limit(get_settings().rate_limit)
async def chat(request: Request, payload: ChatRequest) -> StreamingResponse:
    turns = [{"role": t.role, "content": t.content} for t in payload.messages]
    thread_id = payload.thread_id or uuid4().hex
    return StreamingResponse(
        stream_sse(
            request.app.state, turns, thread_id, locale=payload.locale, embed=payload.embed
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def tool_calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild the turn's tool calls — name, parsed input, result size — from
    the chat-completions messages the tool loop accumulated."""
    results: dict[str, int] = {}
    for message in messages:
        if message.get("role") == "tool":
            results[str(message.get("tool_call_id"))] = len(str(message.get("content") or ""))
    calls: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                {
                    "name": function.get("name") or "",
                    "input": arguments if isinstance(arguments, dict) else {},
                    "result_chars": results.get(str(call.get("id")), 0),
                }
            )
    return calls


def retrieved_chunks_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What search_knowledge handed the model, without the snippet text."""
    search_ids = {
        call.get("id")
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if (call.get("function") or {}).get("name") == "search_knowledge"
    }
    chunks: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "tool" or message.get("tool_call_id") not in search_ids:
            continue
        try:
            payload = json.loads(message.get("content") or "[]")
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        for hit in payload:
            if isinstance(hit, dict):
                chunks.append(
                    {
                        "document": str(hit.get("document") or ""),
                        "heading": str(hit.get("heading") or ""),
                        "score": hit.get("score"),
                    }
                )
    return chunks


def classify_error(exc: BaseException) -> str:
    """Map a failed turn's exception onto the turn_log.error_type vocabulary."""
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return "timeout"
    if isinstance(exc, OpenAICompatError):
        if exc.status == 429:
            return "rate_limited"
        if exc.status is not None and exc.status >= 500:
            return "provider_5xx"
    return "other"


def turn_error_type(
    trace: list[dict[str, Any]], finish_reason: str | None, failure: str | None
) -> str | None:
    """The turn's worst problem: a failed turn's exception class first, then a
    tool that timed out or failed (the answer still went out, but degraded),
    then a truncated answer. None for a clean turn."""
    if failure:
        return failure
    tool_errors = {span.get("error_type") for span in trace if span.get("error_type")}
    if TOOL_ERROR_TIMEOUT in tool_errors:
        return TOOL_ERROR_TIMEOUT
    if TOOL_ERROR_FAILED in tool_errors:
        return TOOL_ERROR_FAILED
    if finish_reason == "length":
        return "truncated"
    return None


def context_chars_from_messages(messages: list[dict[str, Any]]) -> int:
    """Size of the tool results fed back to the model this turn."""
    return sum(
        len(str(message.get("content") or ""))
        for message in messages
        if message.get("role") == "tool"
    )


async def stream_sse(
    app_state: Any,
    turns: list[dict[str, str]],
    thread_id: str,
    *,
    locale: str | None = None,
    embed: bool | None = None,
) -> AsyncIterator[str]:
    agent = app_state.agent
    config = {"configurable": {"thread_id": thread_id}}
    # Gathered along the way for the turn log; the visitor only sees the events.
    outcome = "answered"
    meta: dict[str, Any] | None = None
    tools_used: list[str] = []
    guard_flags: list[str] = []
    citations: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    answer = ""
    # Performance telemetry: wall-clock time to the first visible text, the
    # run trace and per-call breakdown from the final state, the error class.
    started = time.perf_counter()
    ttft_ms: int | None = None
    failure: str | None = None
    trace: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    context_chars: int | None = None
    model_refusal = False
    try:
        async for event in agent.astream({"turns": turns}, config, stream_mode="custom"):
            if event["type"] == "meta":
                meta = event
            elif event["type"] == "tool":
                tools_used.append(event["name"])
            elif event["type"] == "text":
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - started) * 1000)
                answer += event["text"]
            elif event["type"] == "citation":
                citations.append(event["title"])
            yield sse(event)
        snapshot = await agent.aget_state(config)
        guard_flags = list(snapshot.values.get("guard_flags") or [])
        loop_messages = list(snapshot.values.get("loop_messages") or [])
        tool_calls = tool_calls_from_messages(loop_messages)
        retrieved = retrieved_chunks_from_messages(loop_messages)
        context_chars = context_chars_from_messages(loop_messages)
        trace = list(snapshot.values.get("trace") or [])
        model_calls = list(snapshot.values.get("model_calls") or [])
        model_refusal = bool(snapshot.values.get("model_refusal"))
        if snapshot.values.get("refusal"):
            outcome = "refused"
        if snapshot.interrupts:
            # A high-risk tool paused the graph for Ruud's approval. Queue it
            # for the admin endpoint (durable with a database) and tell the visitor.
            outcome = "pending_approval"
            await app_state.approvals.add(thread_id, snapshot.interrupts[0].value or {})
            yield sse({"type": "text", "text": APPROVAL_NOTICE})
            yield sse({"type": "approval", "status": "pending"})
    except Exception as exc:
        logger.exception("Chat stream failed")
        outcome = "error"
        failure = classify_error(exc)
        yield sse({"type": "error", "message": "Unexpected server error."})
        try:
            # Nodes that completed before the failure were checkpointed;
            # keep their spans so the error turn still has a partial trace.
            snapshot = await agent.aget_state(config)
            trace = list(snapshot.values.get("trace") or [])
            model_calls = list(snapshot.values.get("model_calls") or [])
        except Exception:
            logger.debug("No partial state for failed thread %s", thread_id)
    else:
        yield sse({"type": "done"})
    if app_state.recorder is not None:
        # Turn log + budget ledger + guard incidents + unanswered questions;
        # best-effort, never raises.
        await app_state.recorder.record(
            thread_id=thread_id,
            outcome=outcome,
            meta=meta,
            tools_used=tools_used,
            guard_flags=guard_flags,
            visitor_message=turns[-1]["content"],
            answer=answer,
            citations=citations,
            tool_calls=tool_calls,
            retrieved_chunks=retrieved,
            locale=locale,
            embed=embed,
            ttft_ms=ttft_ms,
            error_type=turn_error_type(trace, (meta or {}).get("finishReason"), failure),
            trace=trace,
            model_calls=model_calls,
            context_chars=context_chars,
            model_refusal=model_refusal,
        )
