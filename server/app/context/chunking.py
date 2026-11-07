"""Chunking: every knowledge document becomes heading-scoped paragraph chunks.

A chunk embeds and hashes as `title — heading` plus the paragraph, so it
keeps the context it loses when pulled out of its document, and so editing
one paragraph re-embeds only that chunk.
"""

import hashlib
import re
from dataclasses import dataclass

from .knowledge import KnowledgeDoc

MIN_CHUNK_CHARS = 30

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
