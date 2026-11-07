"""Retrieval over the knowledge base: term overlap as the baseline, cosine
search over embedded chunks when a vector store is wired in.

Sync is incremental: only chunks whose text changed since the last run are
re-embedded, chunks whose text disappeared are removed, and a provider or
model switch re-embeds the lot (hash and model scope every row).

`make_search` is what the search tool runs: semantic when a retriever is
present, falling back to term overlap when it fails or returns nothing.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..storage.models import KnowledgeChunkRow
from .chunking import Chunk, split_into_chunks
from .knowledge import KnowledgeDoc

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 3
MAX_SNIPPET_CHARS = 500


def term_search(docs: list[KnowledgeDoc], query: str) -> list[dict[str, str]]:
    """Rank knowledge paragraphs by term overlap with the query."""
    terms = {term for term in re.findall(r"\w+", query.lower()) if len(term) > 2}
    scored: list[tuple[int, str, str]] = []
    for doc in docs:
        for paragraph in doc.text.split("\n\n"):
            words = set(re.findall(r"\w+", paragraph.lower()))
            if score := len(terms & words):
                scored.append((score, doc.title, paragraph.strip()))
    top = sorted(scored, key=lambda item: -item[0])[:MAX_SEARCH_RESULTS]
    return [{"document": title, "snippet": text[:MAX_SNIPPET_CHARS]} for _, title, text in top]


def make_search(docs: list[KnowledgeDoc], retriever: Any = None) -> Callable[..., Any]:
    """The search callable for the search_knowledge tool: sync term overlap
    without a retriever, async semantic search with one (term overlap as the
    fallback whenever the vector store fails or comes back empty)."""
    if retriever is None:
        return lambda query: term_search(docs, query)

    async def run(query: str) -> list[dict[str, str]]:
        try:
            results = await retriever.search(query)
        except Exception:
            logger.exception("Semantic search failed; falling back to term overlap")
            results = []
        return results or term_search(docs, query)

    return run


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
    """Cosine-distance search over knowledge_chunks. Raises on failure;
    `make_search` catches and falls back to term overlap."""

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
