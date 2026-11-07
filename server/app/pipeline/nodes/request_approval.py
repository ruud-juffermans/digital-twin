"""Node: pause for a human before a write action.

interrupt() checkpoints the graph and ends the visitor's stream with a
pending notice. When Ruud decides via the admin endpoint the graph resumes
here and interrupt() returns the decision.
"""

import logging
from collections.abc import Callable
from typing import Any

from langgraph.types import interrupt

from ..state import AgentState

logger = logging.getLogger(__name__)


def make_request_approval(
    is_high_risk: Callable[[dict[str, Any]], bool],
) -> Callable[[AgentState], Any]:
    async def request_approval(state: AgentState) -> AgentState:
        decision = interrupt(
            {
                "tool_calls": [
                    {"name": call["name"], "input": call["input"]}
                    for call in state.get("pending_tools") or []
                    if is_high_risk(call)
                ],
                "visitor_message": state["turns"][-1]["content"],
            }
        ) or {}
        approved = bool(decision.get("approved"))
        logger.info("High-risk action %s by Ruud", "approved" if approved else "rejected")
        return {"approval": {"decided": True, "approved": approved, "note": decision.get("note")}}

    return request_approval
