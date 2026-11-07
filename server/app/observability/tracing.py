"""The run trace: per-node latency and one span per tool call, streamed to
the client as a diagnostic event and persisted on the turn log."""

import time
from collections.abc import Callable
from typing import Any

# Observability vocabulary for tool spans in the run trace.
TOOL_ERROR_TIMEOUT = "tool_timeout"
TOOL_ERROR_FAILED = "tool_failed"


def traced(name: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a graph node so its latency lands in the run's trace."""

    async def wrapper(state: Any) -> Any:
        started = time.perf_counter()
        updates = await fn(state) or {}
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        base = updates.get("trace", state.get("trace") or [])
        updates["trace"] = [*base, {"node": name, "ms": elapsed_ms}]
        return updates

    return wrapper


def tool_span(name: str, ms: float, *, error_type: str | None) -> dict[str, Any]:
    """A timed tool execution inside the run trace, next to the node timings.
    Names only — tool arguments may carry visitor text and stay out of it."""
    return {"node": f"tool:{name}", "ms": ms, "ok": error_type is None, "error_type": error_type}
