"""The tool contract.

Each tool does exactly one thing, declares a strict JSON Schema
(`additionalProperties: false`, every property required), and executes as
deterministic Python inside a graph node, never in the model layer. A tool
marked high_risk never runs without a human approval.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., Any]  # deterministic, side-effect free
    timeout_s: float = 5.0
    # High-risk tools never run without an explicit human approval.
    high_risk: bool = False


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
