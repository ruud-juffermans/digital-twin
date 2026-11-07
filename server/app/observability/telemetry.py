"""Per-turn telemetry: turn log, model calls, budget ledger, guard incidents,
unanswered questions. Best-effort by design: a telemetry failure must never
break the visitor's chat stream.
"""

import logging
import time
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..storage.models import (
    BudgetLedgerRow,
    GuardIncidentRow,
    ModelCallRow,
    TurnLogRow,
    UnansweredQuestionRow,
)

logger = logging.getLogger(__name__)


def _today() -> str:
    return time.strftime("%Y-%m-%d")



class TurnRecorder:
    """Writes the per-turn telemetry: turn log, budget ledger, guard incidents.

    Best-effort by design — a telemetry failure must never break the visitor's
    chat stream, so every error is logged and swallowed.
    """

    def __init__(
        self,
        sessions: async_sessionmaker,
        provider: str,
        model: str,
        *,
        prompt_version: str | None = None,
        kb_version: str | None = None,
        store_text: bool = True,
    ) -> None:
        self._sessions = sessions
        self._provider = provider
        self._model = model
        self._prompt_version = prompt_version
        self._kb_version = kb_version
        # TEXT_RETENTION_DAYS=0: never store question/answer text at all.
        self._store_text = store_text

    async def record(
        self,
        *,
        thread_id: str,
        outcome: str,
        meta: dict[str, Any] | None,
        tools_used: list[str],
        guard_flags: list[str],
        visitor_message: str,
        answer: str = "",
        citations: list[str] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        retrieved_chunks: list[dict[str, Any]] | None = None,
        locale: str | None = None,
        embed: bool | None = None,
        ttft_ms: int | None = None,
        error_type: str | None = None,
        trace: list[dict[str, Any]] | None = None,
        model_calls: list[dict[str, Any]] | None = None,
        context_chars: int | None = None,
        model_refusal: bool = False,
    ) -> None:
        meta = meta or {}
        cost = float(meta.get("costUsd") or 0.0)
        text_columns: dict[str, Any] = {}
        if self._store_text:
            text_columns = {
                "visitor_message": visitor_message,
                "answer": answer,
                "citations": [{"title": title} for title in citations or []],
                "tool_calls": tool_calls or [],
                "retrieved_chunks": retrieved_chunks or [],
            }
        try:
            async with self._sessions() as session:
                turn = TurnLogRow(
                    thread_id=thread_id,
                    # A fallback turn is billed to the provider that answered.
                    provider=str(meta.get("provider") or self._provider),
                    model=self._model,
                    outcome=outcome,
                    input_tokens=int(meta.get("inputTokens") or 0),
                    output_tokens=int(meta.get("outputTokens") or 0),
                    cached_tokens=int(meta.get("cachedTokens") or 0),
                    cost_usd=cost,
                    latency_ms=int(meta.get("latencyMs") or 0),
                    tools_used=tools_used,
                    guard_flags=guard_flags,
                    prompt_version=self._prompt_version,
                    kb_version=self._kb_version,
                    locale=locale,
                    embed=embed,
                    ttft_ms=ttft_ms,
                    finish_reason=meta.get("finishReason"),
                    retries=int(meta.get("retries") or 0) if meta else None,
                    error_type=error_type,
                    trace=trace,
                    question_chars=len(visitor_message),
                    context_chars=context_chars,
                    answer_chars=len(answer),
                    **text_columns,
                )
                session.add(turn)
                if model_calls:
                    await session.flush()  # assigns turn.id for the call rows
                    for seq, call in enumerate(model_calls):
                        session.add(
                            ModelCallRow(
                                turn_id=turn.id,
                                seq=seq,
                                model=str(call.get("model") or self._model),
                                input_tokens=int(call.get("inputTokens") or 0),
                                output_tokens=int(call.get("outputTokens") or 0),
                                cached_tokens=int(call.get("cachedTokens") or 0),
                                cost_usd=float(call.get("costUsd") or 0.0),
                                latency_ms=int(call.get("latencyMs") or 0),
                                ttft_ms=call.get("ttftMs"),
                                finish_reason=call.get("finishReason"),
                                retries=int(call.get("retries") or 0),
                                tool_call_count=int(call.get("toolCallCount") or 0),
                                output_chars=int(call.get("outputChars") or 0),
                            )
                        )
                if guard_flags:
                    session.add(
                        GuardIncidentRow(
                            thread_id=thread_id, flags=guard_flags, visitor_message=visitor_message
                        )
                    )
                # Content-gap analytics: a guard refusal and an honest
                # model-side "not in my knowledge base" (flagged by the
                # verify node) both count.
                if outcome == "refused":
                    reason = "guard"
                elif outcome == "answered" and model_refusal:
                    reason = "model"
                else:
                    reason = None
                if reason is not None:
                    session.add(
                        UnansweredQuestionRow(
                            thread_id=thread_id,
                            question=visitor_message,
                            answer=answer,
                            reason=reason,
                        )
                    )
                if cost:
                    await self._add_spend(session, cost)
                await session.commit()
        except Exception:
            logger.exception("Failed to record turn telemetry for thread %s", thread_id)

    async def _add_spend(self, session: Any, cost: float) -> None:
        day = _today()
        result = await session.execute(
            update(BudgetLedgerRow)
            .where(BudgetLedgerRow.day == day)
            .values(spent_usd=BudgetLedgerRow.spent_usd + cost)
        )
        if result.rowcount == 0:
            # First turn of the day; a concurrent insert may win the race.
            try:
                async with session.begin_nested():
                    session.add(BudgetLedgerRow(day=day, spent_usd=cost))
            except IntegrityError:
                await session.execute(
                    update(BudgetLedgerRow)
                    .where(BudgetLedgerRow.day == day)
                    .values(spent_usd=BudgetLedgerRow.spent_usd + cost)
                )



async def load_today_spent(sessions: async_sessionmaker) -> float:
    """Seed the in-memory budget tracker after a restart (restart-proof cap)."""
    async with sessions() as session:
        spent = await session.scalar(
            select(BudgetLedgerRow.spent_usd).where(BudgetLedgerRow.day == _today())
        )
    return float(spent or 0.0)

