"""Node: one model round through the gateway.

Text and valid citations stream straight to the visitor; tool calls are
queued for execute_tools; per-round telemetry accumulates into the run's
totals. Past MAX_TOOL_ROUNDS the model is forced to answer in text.
"""

import json
from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer

from ...guardrails.output import valid_citation
from ..state import AgentState

MAX_TOOL_ROUNDS = 3

# Numeric meta fields summed across the tool loop's model rounds. The other
# meta fields (model, ttftMs, finishReason) keep the last round's value; the
# per-round detail lives in `model_calls`.
METERED_FIELDS = ("inputTokens", "outputTokens", "cachedTokens", "latencyMs", "retries")


def _accumulate_meta(totals: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in meta.items() if key != "type"}
    for field in METERED_FIELDS:
        merged[field] = (totals.get(field) or 0) + (meta.get(field) or 0)
    merged["costUsd"] = round((totals.get("costUsd") or 0.0) + (meta.get("costUsd") or 0.0), 5)
    return merged


def _assistant_message(text: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The model's tool-calling turn, echoed back in the wire format."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["input"])},
            }
            for call in calls
        ],
    }


def make_generate(
    get_provider: Callable[[], Any],
    tool_defs: list[dict[str, Any]],
    known_titles: set[str],
) -> Callable[[AgentState], Any]:
    async def generate(state: AgentState) -> AgentState:
        writer = get_stream_writer()
        rounds = state.get("tool_rounds") or 0
        kwargs: dict[str, Any] = {"tools": tool_defs, "extension": state.get("loop_messages") or []}
        if rounds >= MAX_TOOL_ROUNDS:
            # Deterministic budget: past the cap the model must answer in text.
            kwargs["tool_choice"] = "none"

        answer = state.get("last_answer") or ""
        citations = list(state.get("citations") or [])
        flags = list(state.get("guard_flags") or [])
        meta_totals = dict(state.get("loop_meta") or {})
        model_calls = list(state.get("model_calls") or [])
        turn: dict[str, Any] = {}
        async for event in get_provider().stream(state["turns"], **kwargs):
            if event["type"] == "turn":
                turn = event
                continue
            if event["type"] == "meta":
                meta_totals = _accumulate_meta(meta_totals, event)
                # One entry per model round, in order: the per-call breakdown
                # the summed meta cannot give.
                model_calls.append({key: value for key, value in event.items() if key != "type"})
                continue
            if event["type"] == "text":
                answer += event["text"]
            elif event["type"] == "citation":
                if not valid_citation(event.get("title"), known_titles):
                    # Output guard: never surface a citation to a document
                    # that doesn't exist in the knowledge base.
                    flags.append(f"output:unknown-citation:{event.get('title')}")
                    continue
                if event["title"] in citations:
                    continue  # already cited in an earlier round
                citations.append(event["title"])
            writer(event)

        updates: AgentState = {
            "last_answer": answer,
            "citations": citations,
            "guard_flags": flags,
            "loop_meta": meta_totals,
            "model_calls": model_calls,
            "pending_tools": [],
        }
        calls = turn.get("toolCalls") or []
        if calls and rounds < MAX_TOOL_ROUNDS:
            updates["pending_tools"] = calls
            updates["loop_messages"] = [
                *(state.get("loop_messages") or []),
                _assistant_message(turn.get("text") or "", calls),
            ]
        else:
            updates["last_meta"] = meta_totals or None
            if meta_totals:
                writer({"type": "meta", **meta_totals})
        return updates

    return generate
