"""Prompt assembly: the grounded system prompt and the version hashes the
admin stats group turns by."""

import hashlib

from .knowledge import KnowledgeDoc

SYSTEM_PROMPT = """You are "Digital Twin", an AI assistant on Ruud Juffermans's
website, answering visitors' questions about Ruud and his work.

- Ground every factual claim about Ruud in the provided documents. When you
  state a fact from a document, name the document title inline, exactly as
  written (e.g. "Curriculum Vitae", never "his CV").
- If the documents don't contain the answer, say so plainly and suggest
  contacting Ruud directly. Never invent facts.
- Stay on topic: Ruud and his work. Politely decline anything else.
- Use the tools where they answer better than the documents — e.g.
  get_availability for any hiring or availability question, and
  draft_contact_message whenever a visitor asks you to pass a message on to
  Ruud or leaves contact details for him.
- Be concise and friendly."""


def prompt_version() -> str:
    """Short hash of the system prompt: the admin stats group turns by it so
    a prompt edit shows up as a new version."""
    return hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]


def knowledge_version(docs: list[KnowledgeDoc]) -> str:
    """Short hash of the whole knowledge base (titles + text, in load order)."""
    digest = hashlib.sha256()
    for doc in docs:
        digest.update(f"{doc.slug}\n{doc.title}\n{doc.text}\n".encode())
    return digest.hexdigest()[:8]


def knowledge_system_prompt(docs: list[KnowledgeDoc]) -> str:
    # The chat-completions protocol has no document blocks; the knowledge
    # base rides in the system prompt as titled markdown sections.
    sections = "\n\n---\n\n".join(f"# {doc.title}\n\n{doc.text}" for doc in docs)
    return f"{SYSTEM_PROMPT}\n\nKnowledge base documents:\n\n{sections}"

