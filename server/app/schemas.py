from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatTurn] = Field(min_length=1, max_length=40)
    # Stable conversation id for LangGraph checkpointing; the server generates
    # one per request when absent (no cross-turn state in that case).
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # Where the question came from, for the admin stats only: the browser's
    # locale and whether the chat runs embedded on the portfolio (widget).
    locale: str | None = Field(default=None, max_length=16)
    embed: bool | None = None

    @field_validator("messages")
    @classmethod
    def last_turn_is_user(cls, messages: list[ChatTurn]) -> list[ChatTurn]:
        if messages[-1].role != "user":
            raise ValueError("last message must be from the user")
        return messages


class FeedbackRequest(BaseModel):
    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    verdict: Literal["up", "down"]
    question: str | None = Field(default=None, max_length=4000)
    answer: str | None = Field(default=None, max_length=16000)
    comment: str | None = Field(default=None, max_length=1000)
