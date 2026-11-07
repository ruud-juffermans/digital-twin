"""The LangGraph agent: state-controlled orchestration of one chat turn.

Graph shape:

    START → guard_input ─┬→ refuse ─────────────────────────→ END
                         └→ generate ─┬→ request_approval ─┐   (high-risk tool:
                                      │        (interrupt) │    pause for Ruud)
                                      ├→ execute_tools ←───┘
                                      │       ↑└────────┐  (≤ MAX_TOOL_ROUNDS)
                                      │       └─────────┘
                                      └→ verify ─────────→ END

Routing, guarding, tool execution, and verification are deterministic
Python; only `generate` touches a model, through the provider. Nodes emit SSE-ready
event dicts through LangGraph's custom stream writer; chat.py forwards them
to the browser.
"""

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..context.knowledge import KnowledgeDoc
from .budget import BudgetTracker
from .guards import looks_like_refusal, screen_input, valid_citation
from .state import AgentState
from .tools import build_tools, tool_definitions

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3

# Answers longer than this without a single citation get flagged for review.
UNCITED_ANSWER_MIN_CHARS = 200

BUDGET_REFUSAL = (
    "I've reached my usage budget for today — please come back tomorrow, or "
    "contact Ruud directly via the site."
)

# Numeric meta fields summed across the tool loop's model rounds. The other
# meta fields (model, ttftMs, finishReason) keep the last round's value; the
# per-round detail lives in `model_calls`.
METERED_FIELDS = ("inputTokens", "outputTokens", "cachedTokens", "latencyMs", "retries")

# Observability vocabulary for tool spans in the run trace.
TOOL_ERROR_TIMEOUT = "tool_timeout"
TOOL_ERROR_FAILED = "tool_failed"


def _traced(name: str, fn: Callable[[AgentState], Any]) -> Callable[[AgentState], Any]:
    """Record per-node latency into the run's trace."""

    async def wrapper(state: AgentState) -> AgentState:
        started = time.perf_counter()
        updates: AgentState = await fn(state) or {}
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        base = updates.get("trace", state.get("trace") or [])
        updates["trace"] = [*base, {"node": name, "ms": elapsed_ms}]
        return updates

    return wrapper


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


def _tool_message(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


def _tool_span(name: str, ms: float, *, error_type: str | None) -> dict[str, Any]:
    """A timed tool execution inside the run trace, next to the node timings.
    Names only — tool arguments may carry visitor text and stay out of it."""
    return {"node": f"tool:{name}", "ms": ms, "ok": error_type is None, "error_type": error_type}


async def _call_tool(tool: Any, arguments: dict[str, Any]) -> Any:
    """Run a tool: async ones (semantic search) directly, sync ones off-loop."""
    if inspect.iscoroutinefunction(tool.run):
        return await tool.run(**arguments)
    return await asyncio.to_thread(tool.run, **arguments)


def build_agent(
    get_provider: Callable[[], Any],
    docs: list[KnowledgeDoc],
    checkpointer: BaseCheckpointSaver | None = None,
    budget: BudgetTracker | None = None,
    retriever: Any = None,
) -> Any:
    """Compile the agent graph.

    `get_provider` is resolved on every generation so the provider can be
    swapped (tests, future config reload) without rebuilding the graph.
    `retriever` upgrades search_knowledge to semantic search when present.
    """
    known_titles = {doc.title for doc in docs}
    tools_by_name = build_tools(docs, retriever)
    tool_defs = tool_definitions(tools_by_name)

    async def guard_input(state: AgentState) -> AgentState:
        flags: list[str] = []
        refusal: str | None = None
        if budget is not None and budget.exceeded:
            # Fail closed before any tokens are spent.
            refusal = BUDGET_REFUSAL
            flags.append("input:budget")
            logger.warning("Budget guard tripped: $%.2f spent today", budget.spent_usd)
        elif (tripped := screen_input(state["turns"][-1]["content"])) is not None:
            refusal, flag = tripped
            flags.append(flag)
            logger.info("Input guard tripped: %s", flag)
        # Reset per-run state; thread totals (cost, counts) live on.
        return {
            "refusal": refusal,
            "guard_flags": flags,
            "citations": [],
            "last_answer": "",
            "model_refusal": False,
            "last_meta": None,
            "trace": [],
            "model_calls": [],
            "pending_tools": [],
            "loop_messages": [],
            "tool_rounds": 0,
            "loop_meta": None,
            "approval": None,
        }

    def route_after_guard(state: AgentState) -> str:
        return "refuse" if state.get("refusal") else "generate"

    async def refuse(state: AgentState) -> AgentState:
        get_stream_writer()({"type": "text", "text": state["refusal"]})
        return {"message_count": (state.get("message_count") or 0) + 1}

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

    def _high_risk(call: dict[str, Any]) -> bool:
        tool = tools_by_name.get(call["name"])
        return tool is not None and tool.high_risk

    def route_after_generate(state: AgentState) -> str:
        pending = state.get("pending_tools") or []
        if not pending:
            return "verify"
        decided = (state.get("approval") or {}).get("decided")
        if any(_high_risk(call) for call in pending) and not decided:
            return "request_approval"
        return "execute_tools"

    async def request_approval(state: AgentState) -> AgentState:
        # interrupt() pauses the graph and checkpoints it; the visitor's
        # stream ends with a pending notice. When Ruud decides via the admin
        # endpoint, the graph resumes here and interrupt() returns his decision.
        decision = interrupt(
            {
                "tool_calls": [
                    {"name": call["name"], "input": call["input"]}
                    for call in state.get("pending_tools") or []
                    if _high_risk(call)
                ],
                "visitor_message": state["turns"][-1]["content"],
            }
        ) or {}
        approved = bool(decision.get("approved"))
        logger.info("High-risk action %s by Ruud", "approved" if approved else "rejected")
        return {"approval": {"decided": True, "approved": approved, "note": decision.get("note")}}

    async def execute_tools(state: AgentState) -> AgentState:
        writer = get_stream_writer()
        used = list(state.get("tools_used") or [])
        drafts = list(state.get("contact_drafts") or [])
        approval = state.get("approval") or {}
        spans = list(state.get("trace") or [])
        results = []
        for call in state.get("pending_tools") or []:
            name = call["name"]
            writer({"type": "tool", "name": name})
            used.append(name)
            tool = tools_by_name.get(name)
            if tool is None:
                results.append(_tool_message(call, f"Unknown tool: {name}"))
                spans.append(_tool_span(name, 0.0, error_type=TOOL_ERROR_FAILED))
                continue
            if tool.high_risk and not approval.get("approved"):
                # Fail closed: without an explicit approval the action never runs.
                detail = {
                    "status": "declined",
                    "reason": approval.get("note") or "Ruud declined this request.",
                }
                results.append(_tool_message(call, json.dumps(detail)))
                continue
            if tool.high_risk:
                drafts.append({"tool": name, "input": call["input"]})
            started = time.perf_counter()
            error_type: str | None = None
            try:
                value = await asyncio.wait_for(
                    _call_tool(tool, call["input"]), timeout=tool.timeout_s
                )
                results.append(_tool_message(call, json.dumps(value)))
            except TimeoutError:
                error_type = TOOL_ERROR_TIMEOUT
                logger.error("Tool %s timed out after %.1fs", name, tool.timeout_s)
                results.append(_tool_message(call, "Tool timed out."))
            except Exception:
                error_type = TOOL_ERROR_FAILED
                logger.exception("Tool %s failed", name)
                results.append(_tool_message(call, "Tool failed."))
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            spans.append(_tool_span(name, elapsed_ms, error_type=error_type))
        return {
            "trace": spans,
            "pending_tools": [],
            "loop_messages": [*(state.get("loop_messages") or []), *results],
            "tool_rounds": (state.get("tool_rounds") or 0) + 1,
            "tools_used": used,
            "contact_drafts": drafts,
        }

    async def verify(state: AgentState) -> AgentState:
        # Deterministic post-checks and bookkeeping; the run's per-node trace
        # goes out as a diagnostic SSE event.
        cost = (state.get("last_meta") or {}).get("costUsd") or 0.0
        if budget is not None and cost:
            budget.add(cost)
        flags = list(state.get("guard_flags") or [])
        answer = state.get("last_answer") or ""
        refusal = looks_like_refusal(answer)
        if (
            len(answer) >= UNCITED_ANSWER_MIN_CHARS
            and not state.get("citations")
            and not state.get("tool_rounds")  # tool results are grounding too
            and not refusal
        ):
            # Groundedness check: a substantive factual answer with zero
            # citations is a hallucination risk — flag it for review.
            flags.append("output:uncited-answer")
            logger.warning("Uncited answer flagged (%d chars)", len(answer))
        get_stream_writer()(
            {
                "type": "trace",
                "nodes": [*(state.get("trace") or []), {"node": "verify", "ms": None}],
                "guardFlags": flags,
            }
        )
        return {
            "guard_flags": flags,
            "model_refusal": refusal,
            "message_count": (state.get("message_count") or 0) + 1,
            "total_cost_usd": round((state.get("total_cost_usd") or 0.0) + cost, 5),
        }

    graph = StateGraph(AgentState)
    graph.add_node("guard_input", _traced("guard_input", guard_input))
    graph.add_node("refuse", _traced("refuse", refuse))
    graph.add_node("generate", _traced("generate", generate))
    graph.add_node("request_approval", _traced("request_approval", request_approval))
    graph.add_node("execute_tools", _traced("execute_tools", execute_tools))
    graph.add_node("verify", _traced("verify", verify))
    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", route_after_guard, ["refuse", "generate"])
    graph.add_conditional_edges(
        "generate", route_after_generate, ["request_approval", "execute_tools", "verify"]
    )
    graph.add_edge("request_approval", "execute_tools")
    graph.add_edge("execute_tools", "generate")
    graph.add_edge("refuse", END)
    graph.add_edge("verify", END)
    return graph.compile(checkpointer=checkpointer)
