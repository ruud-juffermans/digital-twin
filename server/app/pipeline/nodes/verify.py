"""Node: deterministic post-checks and bookkeeping after the final model
round. Output guards can only flag here, never block: the text has already
streamed. The run's trace goes out as a diagnostic event."""

import logging
from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer

from ...actions.budget import BudgetTracker
from ...guardrails.output import looks_like_refusal, uncited_answer
from ..state import AgentState

logger = logging.getLogger(__name__)


def make_verify(budget: BudgetTracker | None) -> Callable[[AgentState], Any]:
    async def verify(state: AgentState) -> AgentState:
        cost = (state.get("last_meta") or {}).get("costUsd") or 0.0
        if budget is not None and cost:
            budget.add(cost)
        flags = list(state.get("guard_flags") or [])
        answer = state.get("last_answer") or ""
        refusal = looks_like_refusal(answer)
        if uncited_answer(
            answer,
            citations=list(state.get("citations") or []),
            tool_rounds=state.get("tool_rounds") or 0,
        ):
            # A substantive factual answer with zero citations is a
            # hallucination risk — flag it for review.
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

    return verify
