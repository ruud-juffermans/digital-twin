"""The LangGraph agent: state-controlled orchestration of one chat turn.

Graph shape:

    START → guard_input ─┬→ refuse ─────────────────────────→ END
                         └→ generate ─┬→ request_approval ─┐   (high-risk tool:
                                      │        (interrupt) │    pause for Ruud)
                                      ├→ execute_tools ←───┘
                                      │       ↑└────────┐  (≤ MAX_TOOL_ROUNDS)
                                      │       └─────────┘
                                      └→ verify ─────────→ END

Routing, guarding, tool execution, and verification are deterministic
Python; only `generate` touches a model, through the provider. Nodes emit SSE-ready
event dicts through LangGraph's custom stream writer; chat.py forwards them
to the browser.
"""

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..actions.budget import BudgetTracker
from ..actions.tools.base import tool_definitions
from ..actions.tools.registry import build_tools
from ..context.knowledge import KnowledgeDoc
from ..context.retrieval import make_search
from ..observability.tracing import traced
from .nodes.execute_tools import make_execute_tools
from .nodes.generate import make_generate
from .nodes.guard_input import make_guard_input
from .nodes.refuse import refuse
from .nodes.request_approval import make_request_approval
from .nodes.verify import make_verify
from .state import AgentState


def build_agent(
    get_provider: Callable[[], Any],
    docs: list[KnowledgeDoc],
    checkpointer: BaseCheckpointSaver | None = None,
    budget: BudgetTracker | None = None,
    retriever: Any = None,
) -> Any:
    """Compile the agent graph.

    `get_provider` is resolved on every generation so the provider can be
    swapped (tests, future config reload) without rebuilding the graph.
    `retriever` upgrades search_knowledge to semantic search when present.
    """
    known_titles = {doc.title for doc in docs}
    tools_by_name = build_tools(make_search(docs, retriever))
    tool_defs = tool_definitions(tools_by_name)


    def is_high_risk(call: dict[str, Any]) -> bool:
        tool = tools_by_name.get(call["name"])
        return tool is not None and tool.high_risk

    def route_after_guard(state: AgentState) -> str:
        return "refuse" if state.get("refusal") else "generate"

    def route_after_generate(state: AgentState) -> str:
        pending = state.get("pending_tools") or []
        if not pending:
            return "verify"
        decided = (state.get("approval") or {}).get("decided")
        if any(is_high_risk(call) for call in pending) and not decided:
            return "request_approval"
        return "execute_tools"

    guard_input = make_guard_input(budget)
    generate = make_generate(get_provider, tool_defs, known_titles)
    request_approval = make_request_approval(is_high_risk)
    execute_tools = make_execute_tools(tools_by_name)
    verify = make_verify(budget)

    graph = StateGraph(AgentState)
    graph.add_node("guard_input", traced("guard_input", guard_input))
    graph.add_node("refuse", traced("refuse", refuse))
    graph.add_node("generate", traced("generate", generate))
    graph.add_node("request_approval", traced("request_approval", request_approval))
    graph.add_node("execute_tools", traced("execute_tools", execute_tools))
    graph.add_node("verify", traced("verify", verify))
    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", route_after_guard, ["refuse", "generate"])
    graph.add_conditional_edges(
        "generate", route_after_generate, ["request_approval", "execute_tools", "verify"]
    )
    graph.add_edge("request_approval", "execute_tools")
    graph.add_edge("execute_tools", "generate")
    graph.add_edge("refuse", END)
    graph.add_edge("verify", END)
    return graph.compile(checkpointer=checkpointer)
