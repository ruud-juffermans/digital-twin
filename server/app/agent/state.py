"""Typed state flowing through the LangGraph agent.

One AgentState per conversation thread. With a checkpointer attached it is
persisted per `thread_id`, so the running totals accumulate across turns.
Per-run fields are reset by `guard_input` at the start of every run.
"""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # The full conversation for this turn (the client sends complete history).
    turns: list[dict[str, str]]
    # Set by guard_input; non-None routes to the deterministic refuse node.
    refusal: str | None
    # Guards that tripped during this run, e.g. "input:injection".
    guard_flags: list[str]
    # Document titles cited in this run's answer.
    citations: list[str]
    # Full answer text of this run (for deterministic output checks).
    last_answer: str
    # Set by verify: the answer reads as an honest "not in my knowledge base"
    # (content-gap analytics, not a guard trip).
    model_refusal: bool
    # Usage/cost/latency telemetry of this run's generation.
    last_meta: dict[str, Any] | None
    # Per-node latency of this run: [{"node": name, "ms": float}], plus one
    # {"node": "tool:<name>", "ms", "ok", "error_type"} span per tool call.
    trace: list[dict[str, Any]]
    # One meta dict per model round of this run (tokens, cost, latency,
    # ttftMs, finishReason, retries) — the per-call breakdown.
    model_calls: list[dict[str, Any]]
    # Tool loop (reset per run):
    # calls from the last model turn awaiting deterministic execution,
    # as {"id", "name", "input"}.
    pending_tools: list[dict[str, Any]]
    # Chat-completions messages accumulated this run: the assistant turns
    # carrying tool_calls and the role:"tool" results that answer them.
    loop_messages: list[dict[str, Any]]
    tool_rounds: int
    # Usage/cost totals aggregated across the run's model rounds.
    loop_meta: dict[str, Any] | None
    # Every tool the model called on this thread (checkpointed audit trail).
    tools_used: list[str]
    # Ruud's decision on the pending high-risk action, set on graph resume:
    # {"decided": bool, "approved": bool, "note": str | None}.
    approval: dict[str, Any] | None
    # Approved contact drafts, persisted for the thread's lifetime.
    contact_drafts: list[dict[str, Any]]
    # Accumulated across the thread's lifetime (checkpointed).
    total_cost_usd: float
    message_count: int
