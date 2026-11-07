"""Read-only: look a fact up in the knowledge base.

The search itself is injected (term overlap, or semantic with a fallback) so
this module knows the tool contract and nothing about where the text lives.
"""

from collections.abc import Callable
from typing import Any

from .base import Tool


def search_knowledge(search: Callable[..., Any]) -> Tool:
    return Tool(
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
    )
