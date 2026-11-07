"""Visitor feedback: thumbs up or down per answer, with an optional comment."""

from sqlalchemy.ext.asyncio import async_sessionmaker

from ..storage.models import FeedbackRow


async def record_feedback(
    sessions: async_sessionmaker,
    *,
    thread_id: str,
    verdict: str,
    question: str | None,
    answer: str | None,
    comment: str | None,
) -> None:
    async with sessions() as session:
        session.add(
            FeedbackRow(
                thread_id=thread_id,
                verdict=verdict,
                question=question,
                answer=answer,
                comment=comment,
            )
        )
        await session.commit()

