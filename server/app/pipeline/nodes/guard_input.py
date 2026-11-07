"""Node: run the input guards before a single token is spent, and reset the
per-run state. The budget guard goes first: an exhausted budget is the one
failure that should never reach the model."""

import logging
from collections.abc import Callable
from typing import Any

from ...actions.budget import BudgetTracker
from ...guardrails.input import screen_input
from ...guardrails.policies import BUDGET_REFUSAL
from ..state import AgentState

logger = logging.getLogger(__name__)


def make_guard_input(budget: BudgetTracker | None) -> Callable[[AgentState], Any]:
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

    return guard_input
