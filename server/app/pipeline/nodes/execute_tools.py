"""Node: run the model's tool calls as deterministic Python.

Every call is timed into the run trace. A high-risk tool without an explicit
approval is declined, never run. Timeouts and exceptions become tool
messages so the model can still answer.
"""

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer

from ...actions.tools.base import Tool
from ...observability.tracing import TOOL_ERROR_FAILED, TOOL_ERROR_TIMEOUT, tool_span
from ..state import AgentState

logger = logging.getLogger(__name__)


def _tool_message(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


async def _call_tool(tool: Any, arguments: dict[str, Any]) -> Any:
    """Run a tool: async ones (semantic search) directly, sync ones off-loop."""
    if inspect.iscoroutinefunction(tool.run):
        return await tool.run(**arguments)
    return await asyncio.to_thread(tool.run, **arguments)


def make_execute_tools(tools_by_name: dict[str, Tool]) -> Callable[[AgentState], Any]:
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
                spans.append(tool_span(name, 0.0, error_type=TOOL_ERROR_FAILED))
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
            spans.append(tool_span(name, elapsed_ms, error_type=error_type))
        return {
            "trace": spans,
            "pending_tools": [],
            "loop_messages": [*(state.get("loop_messages") or []), *results],
            "tool_rounds": (state.get("tool_rounds") or 0) + 1,
            "tools_used": used,
            "contact_drafts": drafts,
        }

    return execute_tools
