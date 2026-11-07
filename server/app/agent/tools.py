"""Narrow-scoped tools.

Each tool does exactly one thing, declares a strict JSON Schema
(`additionalProperties: false`, every property required), and executes as
deterministic Python inside a graph node — never in the model layer. The
high-risk `draft_contact_message` tool never runs without a human approval;
nothing here can act on the outside world.

`search_knowledge` is semantic when a retriever is wired in (Postgres +
pgvector) and falls back to term overlap without one — or whenever the
vector search fails or comes back empty.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..context.knowledge import KnowledgeDoc
from ..context.retrieval import make_search

logger = logging.getLogger(__name__)

# Keep in sync with the knowledge modules — this is the structured source the
# model can quote exactly instead of paraphrasing prose.
AVAILABILITY = {
    "status": "open",
    "available_from": "now",
    "looking_for": [
        "fixed-scope data & AI consulting engagements",
        "data platforms, dashboards, AI prototypes, data quality monitoring",
    ],
    "location": "Amsterdam — working with organisations across the Netherlands",
    "contact": "via the contact form on the website",
    "note": "This assistant cannot make commitments; for anything contractual, contact Ruud.",
}

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., Any]  # deterministic, side-effect free
    timeout_s: float = 5.0
    # High-risk tools never run without an explicit human approval.
    high_risk: bool = False


def build_tools(docs: list[KnowledgeDoc], retriever: Any = None) -> dict[str, Tool]:
    search = make_search(docs, retriever)
    return {
        tool.name: tool
        for tool in (
            Tool(
                name="search_knowledge",
                description=(
                    "Search Ruud's knowledge base (CV, projects, about) for paragraphs "
                    "matching a query. Use when you need to double-check a specific fact."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keywords to search for."}
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                run=search,
            ),
            Tool(
                name="get_availability",
                description=(
                    "Get Ruud's current availability for roles and freelance work as "
                    "structured data. Use for any question about hiring or availability."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                run=lambda: AVAILABILITY,
            ),
            Tool(
                name="draft_contact_message",
                description=(
                    "Submit a message from the visitor to Ruud (hiring inquiry, "
                    "collaboration, question). Call it whenever the visitor asks you to "
                    "send, pass on, or forward something to Ruud, or leaves contact "
                    "details for him. Requires Ruud's personal approval before it is "
                    "recorded — tell the visitor it was submitted for approval."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Short subject line."},
                        "message": {"type": "string", "description": "The visitor's message."},
                        "sender_contact": {
                            "type": "string",
                            "description": "How Ruud can reach the visitor (email or similar).",
                        },
                    },
                    "required": ["subject", "message", "sender_contact"],
                    "additionalProperties": False,
                },
                run=lambda subject, message, sender_contact: {
                    "status": "recorded",
                    "subject": subject,
                    "message": message,
                    "sender_contact": sender_contact,
                },
                high_risk=True,
            ),
        )
    }


def tool_definitions(tools: dict[str, Tool]) -> list[dict[str, Any]]:
    """Tool definitions in the chat-completions `tools` wire format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools.values()
    ]
