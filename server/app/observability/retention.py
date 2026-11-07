"""Text retention: visitor text is kept TEXT_RETENTION_DAYS, numeric
telemetry forever."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..storage.models import FeedbackRow, TurnLogRow

logger = logging.getLogger(__name__)

RETENTION_INTERVAL_S = 24 * 60 * 60


async def purge_expired_text(sessions: async_sessionmaker, retention_days: int) -> int:
    """Null the free-text columns of turns and feedback older than the
    retention window (PRD: visitor text is kept TEXT_RETENTION_DAYS, numeric
    telemetry forever). Returns the number of turn rows cleared."""
    cutoff = datetime.now(UTC) - timedelta(days=max(retention_days, 0))
    async with sessions() as session:
        turns = await session.execute(
            update(TurnLogRow)
            .where(TurnLogRow.created_at < cutoff, TurnLogRow.visitor_message.is_not(None))
            .values(
                visitor_message=None,
                answer=None,
                citations=None,
                tool_calls=None,
                retrieved_chunks=None,
            )
        )
        await session.execute(
            update(FeedbackRow)
            .where(
                FeedbackRow.created_at < cutoff,
                (FeedbackRow.question.is_not(None))
                | (FeedbackRow.answer.is_not(None))
                | (FeedbackRow.comment.is_not(None)),
            )
            .values(question=None, answer=None, comment=None)
        )
        await session.commit()
    return turns.rowcount or 0



async def retention_loop(sessions: Any, retention_days: int) -> None:
    """Clear expired visitor text once at startup, then daily."""
    while True:
        try:
            cleared = await purge_expired_text(sessions, retention_days)
            if cleared:
                logger.info("Retention: cleared text on %d turn(s) older than %d days",
                            cleared, retention_days)
        except Exception:
            logger.exception("Retention job failed")
        await asyncio.sleep(RETENTION_INTERVAL_S)
