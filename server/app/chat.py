import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_settings
from .db import record_feedback
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
    try:
        async for event in agent.astream({"turns": turns}, config, stream_mode="custom"):
            if event["type"] == "meta":
                meta = event
            elif event["type"] == "tool":
                tools_used.append(event["name"])
            elif event["type"] == "text":
                answer += event["text"]
            elif event["type"] == "citation":
                citations.append(event["title"])
            yield sse(event)
        snapshot = await agent.aget_state(config)
        guard_flags = list(snapshot.values.get("guard_flags") or [])
        loop_messages = list(snapshot.values.get("loop_messages") or [])
        tool_calls = tool_calls_from_messages(loop_messages)
        retrieved = retrieved_chunks_from_messages(loop_messages)
        if snapshot.values.get("refusal"):
            outcome = "refused"
        if snapshot.interrupts:
            # A high-risk tool paused the graph for Ruud's approval. Queue it
            # for the admin endpoint (durable with a database) and tell the visitor.
            outcome = "pending_approval"
            await app_state.approvals.add(thread_id, snapshot.interrupts[0].value or {})
            yield sse({"type": "text", "text": APPROVAL_NOTICE})
            yield sse({"type": "approval", "status": "pending"})
    except Exception:
        logger.exception("Chat stream failed")
        outcome = "error"
        yield sse({"type": "error", "message": "Unexpected server error."})
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
        )
