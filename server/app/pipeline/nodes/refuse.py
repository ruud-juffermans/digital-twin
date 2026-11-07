"""Node: deliver a guard's refusal as the whole answer, no model involved."""

from langgraph.config import get_stream_writer

from ..state import AgentState


async def refuse(state: AgentState) -> AgentState:
    get_stream_writer()({"type": "text", "text": state["refusal"]})
    return {"message_count": (state.get("message_count") or 0) + 1}
