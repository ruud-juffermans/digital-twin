"""Every tool the model may call, by name. Adding a tool means adding a
module next to this one and listing it here; the graph reads the read/write
flag off the Tool itself."""

from collections.abc import Callable
from typing import Any

from .base import Tool
from .draft_contact_message import draft_contact_message
from .get_availability import get_availability
from .search_knowledge import search_knowledge


def build_tools(search: Callable[..., Any]) -> dict[str, Tool]:
    tools = (search_knowledge(search), get_availability(), draft_contact_message())
    return {tool.name: tool for tool in tools}
