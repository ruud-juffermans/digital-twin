"""Semantic retrieval over the knowledge base.

Chunking + sync: every knowledge document is split into heading-scoped
paragraph chunks, hashed, and embedded into the knowledge_chunks table. Sync
is incremental — only chunks whose text changed since the last run are
re-embedded, chunks whose text disappeared are removed, and a
provider/model switch re-embeds the lot (hash and model scope every row).

Search: `SemanticRetriever.search` embeds the query and ranks chunks by
cosine distance in Postgres. The term-overlap fallback lives in
agent/tools.py, which owns the tool contract.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from .knowledge import KnowledgeDoc
from .storage.models import KnowledgeChunkRow

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 30
MAX_SEARCH_RESULTS = 3
MAX_SNIPPET_CHARS = 500

_ANY_HEADING = re.compile(r"^#{1,6}\s+")
# Only ## and deeper become chunk context — an H1 would repeat the doc
# title, which is already in every chunk's embed prefix.
_SECTION_HEADING = re.compile(r"^#{2,6}\s+(.+)$")


@dataclass(frozen=True)
class Chunk:
    doc_slug: str
    doc_title: str
    heading: str
    index: int
    content: str

    @property
    def embed_text(self) -> str:
        """What gets embedded and hashed: title + heading give the paragraph
        the context it loses when pulled out of its document."""
        prefix = f"{self.doc_title} — {self.heading}" if self.heading else self.doc_title
        return f"{prefix}\n\n{self.content}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embed_text.encode()).hexdigest()


def split_into_chunks(doc: KnowledgeDoc) -> list[Chunk]:
    """Heading-scoped paragraph chunks; tiny fragments dropped."""
    chunks: list[Chunk] = []
    heading = ""
    for block in doc.text.split("\n\n"):
        lines = []
        for line in block.strip().splitlines():
            if (match := _SECTION_HEADING.match(line)) is not None:
                heading = match.group(1).strip()
            elif _ANY_HEADING.match(line) is None:
                lines.append(line)
        content = "\n".join(lines).strip()
        if len(content) >= MIN_CHUNK_CHARS:
            chunks.append(
                Chunk(
                    doc_slug=doc.slug,
                    doc_title=doc.title,
                    heading=heading,
                    index=len(chunks),
                    content=content,
                )
            )
    return chunks


async def sync_knowledge_chunks(
    sessions: async_sessionmaker, embedder: Any, docs: list[KnowledgeDoc]
) -> dict[str, int]:
    """Bring knowledge_chunks in line with `docs`, embedding only what changed."""
    chunks: dict[tuple[str, str], Chunk] = {}
    for doc in docs:
        for chunk in split_into_chunks(doc):
            chunks.setdefault((doc.slug, chunk.content_hash), chunk)

    async with sessions() as session:
        rows = (await session.execute(select(KnowledgeChunkRow))).scalars().all()
        current = {
            (row.doc_slug, row.content_hash): row
            for row in rows
            if row.embedding_model == embedder.model_id
        }
        # Rows from another embedding model live in a different vector space;
        # rows whose text disappeared are stale either way.
        obsolete = [row for row in rows if row.embedding_model != embedder.model_id] + [
            row for key, row in current.items() if key not in chunks
        ]
        new_chunks = [chunk for key, chunk in chunks.items() if key not in current]

        vectors = await embedder.embed([chunk.embed_text for chunk in new_chunks])
        for chunk, vector in zip(new_chunks, vectors, strict=True):
            session.add(
                KnowledgeChunkRow(
                    doc_slug=chunk.doc_slug,
                    doc_title=chunk.doc_title,
                    heading=chunk.heading,
                    chunk_index=chunk.index,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    embedding=vector,
                    embedding_model=embedder.model_id,
                )
            )
        for row in obsolete:
            await session.delete(row)
        await session.commit()

    return {
        "embedded": len(new_chunks),
        "kept": len(chunks) - len(new_chunks),
        "deleted": len(obsolete),
    }


class SemanticRetriever:
    """Cosine-distance search over knowledge_chunks. Raises on failure — the
    search tool in agent/tools.py catches and falls back to term overlap."""

    def __init__(self, embedder: Any, sessions: async_sessionmaker) -> None:
        self._embedder = embedder
        self._sessions = sessions

    async def search(self, query: str) -> list[dict[str, str]]:
        [vector] = await self._embedder.embed([query])
        literal = "[" + ",".join(f"{value:g}" for value in vector) + "]"
        sql = text(
            "SELECT doc_title, heading, content FROM knowledge_chunks "
            "WHERE embedding_model = :model "
            "ORDER BY embedding <=> CAST(:vec AS vector) LIMIT :k"
        )
        async with self._sessions() as session:
            rows = await session.execute(
                sql, {"model": self._embedder.model_id, "vec": literal, "k": MAX_SEARCH_RESULTS}
            )
            return [
                {
                    "document": doc_title,
                    "heading": heading,
                    "snippet": content[:MAX_SNIPPET_CHARS],
                }
                for doc_title, heading, content in rows
            ]
