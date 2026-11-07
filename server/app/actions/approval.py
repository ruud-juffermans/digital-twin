"""Human-in-the-loop gate for write actions.

A high-risk tool call interrupts the graph; the thread sits checkpointed in
the approval queue until Ruud decides. The stores below hold that queue
(Postgres-backed with a database, in-memory otherwise) and archive the
approved drafts. The checkpoints stay the source of truth for the paused run.
"""

from typing import Any, Protocol

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..storage.models import ContactMessageRow, PendingApprovalRow


class ApprovalStore(Protocol):
    async def add(self, thread_id: str, payload: dict[str, Any]) -> None: ...
    async def get(self, thread_id: str) -> dict[str, Any] | None: ...
    async def list(self) -> list[dict[str, Any]]: ...
    async def remove(self, thread_id: str) -> None: ...


class InMemoryApprovalStore:
    """Fallback without a database; the checkpoints stay the source of truth."""

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, Any]] = {}

    async def add(self, thread_id: str, payload: dict[str, Any]) -> None:
        self._pending[thread_id] = {"thread_id": thread_id, **payload}

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._pending.get(thread_id)

    async def list(self) -> list[dict[str, Any]]:
        return list(self._pending.values())

    async def remove(self, thread_id: str) -> None:
        self._pending.pop(thread_id, None)


class PostgresApprovalStore:
    def __init__(self, sessions: async_sessionmaker) -> None:
        self._sessions = sessions

    async def add(self, thread_id: str, payload: dict[str, Any]) -> None:
        async with self._sessions() as session:
            await session.merge(PendingApprovalRow(thread_id=thread_id, payload=payload))
            await session.commit()

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        async with self._sessions() as session:
            row = await session.get(PendingApprovalRow, thread_id)
        return None if row is None else {"thread_id": row.thread_id, **row.payload}

    async def list(self) -> list[dict[str, Any]]:
        async with self._sessions() as session:
            query = select(PendingApprovalRow).order_by(PendingApprovalRow.created_at)
            rows = (await session.execute(query)).scalars().all()
        return [{"thread_id": row.thread_id, **row.payload} for row in rows]

    async def remove(self, thread_id: str) -> None:
        async with self._sessions() as session:
            await session.execute(
                delete(PendingApprovalRow).where(PendingApprovalRow.thread_id == thread_id)
            )
            await session.commit()



async def archive_contact_messages(
    sessions: async_sessionmaker,
    *,
    thread_id: str,
    tool_calls: list[dict[str, Any]],
    note: str | None,
) -> int:
    """Archive the draft_contact_message payloads of an approved action."""
    drafts = [
        call.get("input") or {}
        for call in tool_calls
        if call.get("name") == "draft_contact_message"
    ]
    async with sessions() as session:
        for draft in drafts:
            session.add(
                ContactMessageRow(
                    thread_id=thread_id,
                    subject=str(draft.get("subject") or ""),
                    message=str(draft.get("message") or ""),
                    sender_contact=str(draft.get("sender_contact") or ""),
                    note=note,
                )
            )
        await session.commit()
    return len(drafts)

